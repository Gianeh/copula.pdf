"""
Ablation study: isolamento dei contributi di trasformazione input e N.

Design fattoriale 2×2 su ciascuna delle 2 baseline del checkpoint 1:
  - Naive:  64×2, softplus, λ=100, 5000 epoche
  - Best:   128×3, softplus, λ=10, 5000 epoche

Fattori: N ∈ {2000, 10000}, input_transform ∈ {False, True}.
Tutto il resto è fisso (softplus, nessun scheduler, nessun clip).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch

from copula import (
    true_copula_on_grid, generate_data,
    CopulaDensityNet, train, predict_on_grid, kl_divergence
)
from experiment_utils import (
    init_run, save_current_figure, save_model_state, save_results, set_global_seed
)

RHO = 0.7
GRID_EVAL = 200
SEED = 42
WRITE_LEGACY_ARTIFACTS = os.getenv('WRITE_LEGACY_ARTIFACTS', '0') == '1'

run_root, plots_dir, legacy_plots_dir, _ = init_run(
    __file__,
    seed=SEED,
    config={
        'rho': RHO,
        'grid_eval': GRID_EVAL,
    },
    legacy_artifacts=WRITE_LEGACY_ARTIFACTS,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Run directory: {run_root}\n")

U_grid, V_grid, c_true = true_copula_on_grid(RHO, GRID_EVAL)
eps = 0.01
du = (1 - 2 * eps) / (GRID_EVAL - 1)

# --- Configurazioni ---
configs = [
    # Gruppo 1: Naive (64×2, λ=100)
    {'name': 'naive',          'group': 'Naive', 'hidden': 64,  'layers': 2, 'lam': 100,
     'N': 2000,  'transform': False, 'epochs': 5000, 'lr': 1e-3},
    {'name': 'naive+Φ⁻¹',      'group': 'Naive', 'hidden': 64,  'layers': 2, 'lam': 100,
     'N': 2000,  'transform': True,  'epochs': 5000, 'lr': 1e-3},
    {'name': 'naive+10k',      'group': 'Naive', 'hidden': 64,  'layers': 2, 'lam': 100,
     'N': 10000, 'transform': False, 'epochs': 5000, 'lr': 1e-3},
    {'name': 'naive+Φ⁻¹+10k',  'group': 'Naive', 'hidden': 64,  'layers': 2, 'lam': 100,
     'N': 10000, 'transform': True,  'epochs': 5000, 'lr': 1e-3},

    # Gruppo 2: Best (128×3, λ=10)
    {'name': 'best',           'group': 'Best',  'hidden': 128, 'layers': 3, 'lam': 10,
     'N': 2000,  'transform': False, 'epochs': 5000, 'lr': 1e-3},
    {'name': 'best+Φ⁻¹',       'group': 'Best',  'hidden': 128, 'layers': 3, 'lam': 10,
     'N': 2000,  'transform': True,  'epochs': 5000, 'lr': 1e-3},
    {'name': 'best+10k',       'group': 'Best',  'hidden': 128, 'layers': 3, 'lam': 10,
     'N': 10000, 'transform': False, 'epochs': 5000, 'lr': 1e-3},
    {'name': 'best+Φ⁻¹+10k',   'group': 'Best',  'hidden': 128, 'layers': 3, 'lam': 10,
     'N': 10000, 'transform': True,  'epochs': 5000, 'lr': 1e-3},
]

results = []

for cfg in configs:
    name = cfg['name']
    tr = 'Sì' if cfg['transform'] else 'No'
    print(f"\n{'='*60}")
    print(f"  {name}  ({cfg['group']})")
    print(f"  {cfg['hidden']}×{cfg['layers']}, λ={cfg['lam']}, N={cfg['N']}, Φ⁻¹={tr}")
    print(f"{'='*60}")

    X, Y, U, V = generate_data(RHO, cfg['N'], seed=SEED)
    set_global_seed(SEED + len(results) + 1)
    model = CopulaDensityNet(hidden=cfg['hidden'], layers=cfg['layers'],
                             output_act='softplus',
                             input_transform=cfg['transform']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametri: {n_params}")

    train(model, U, V, device,
          epochs=cfg['epochs'], lr=cfg['lr'], lam=cfg['lam'],
          grid_size=50, print_every=2500)

    _, _, c_pred = predict_on_grid(model, device, GRID_EVAL)
    kl = kl_divergence(c_true, c_pred, du)
    integ = np.trapz(np.trapz(c_pred, dx=du, axis=1), dx=du)
    err = np.abs(c_true - c_pred)

    # Ricostruzione PDF congiunta
    grid_j = 150
    x_range = np.linspace(-3.5, 3.5, grid_j)
    y_range = np.linspace(-3.5, 3.5, grid_j)
    Xg, Yg = np.meshgrid(x_range, y_range, indexing='ij')
    u_vals = stats.norm.cdf(Xg)
    v_vals = stats.norm.cdf(Yg)
    uv_t = torch.tensor(np.column_stack((u_vals.ravel(), v_vals.ravel())),
                        dtype=torch.float32, device=device)
    with torch.no_grad():
        c_hat = model(uv_t).cpu().numpy().reshape(grid_j, grid_j)
    cov = [[1, RHO], [RHO, 1]]
    pos = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    f_true = stats.multivariate_normal.pdf(pos, mean=[0, 0], cov=cov).reshape(grid_j, grid_j)
    f_recon = c_hat * stats.norm.pdf(Xg) * stats.norm.pdf(Yg)
    dx_j = (x_range[-1] - x_range[0]) / (grid_j - 1)
    l1_joint = np.trapz(np.trapz(np.abs(f_true - f_recon), dx=dx_j, axis=1), dx=dx_j)

    results.append({**cfg, 'kl': kl, 'integ': integ,
                    'err_max': err.max(), 'err_mean': err.mean(),
                    'l1_joint': l1_joint, 'c_pred': c_pred, 'model': model})

    print(f"\n  KL={kl:.6f}  ∫∫ĉ={integ:.4f}  err_max={err.max():.3f}  "
          f"err_mean={err.mean():.4f}  L1_joint={l1_joint:.4f}")

# --- Tabella ---
print(f"\n{'='*110}")
print(f"{'Config':<20} {'Group':<6} {'H×L':>5} {'λ':>4} {'N':>6} {'Φ⁻¹':>4} "
      f"{'KL':>10} {'∫∫ĉ':>7} {'err_max':>8} {'err_mean':>9} {'L1_joint':>9}")
print("-" * 110)
for r in results:
    hl = f"{r['hidden']}x{r['layers']}"
    tr = 'Y' if r['transform'] else 'N'
    print(f"{r['name']:<20} {r['group']:<6} {hl:>5} {r['lam']:>4} {r['N']:>6} {tr:>4} "
          f"{r['kl']:>10.6f} {r['integ']:>7.4f} {r['err_max']:>8.3f} "
          f"{r['err_mean']:>9.4f} {r['l1_joint']:>9.4f}")

# --- Bar chart raggruppato ---
fig, ax = plt.subplots(figsize=(12, 5))

groups = ['Naive', 'Best']
labels = ['baseline', '+Φ⁻¹', '+10k', '+Φ⁻¹+10k']
colors = ['#cccccc', '#5dade2', '#f5b041', '#2ecc71']
x = np.arange(len(labels))
width = 0.35

for i, group in enumerate(groups):
    group_results = [r for r in results if r['group'] == group]
    kls = [r['kl'] for r in group_results]
    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, kls, width, label=group, color=colors, edgecolor='black',
                  linewidth=0.5, alpha=0.85 if i == 0 else 1.0)
    if i == 0:
        for bar in bars:
            bar.set_hatch('//')
    for bar, kl_val in zip(bars, kls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{kl_val:.4f}', ha='center', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('KL Divergence')
ax.set_title('Ablation study: contributo di trasformazione Φ⁻¹ e N campioni')
ax.legend(title='Config base', loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
save_current_figure(plots_dir, 'ablation_comparison.png', legacy_plots_dir)
plt.close()

# --- Contour per il best complessivo ---
best = min(results, key=lambda r: r['kl'])
print(f"\nBest complessivo: {best['name']} (KL={best['kl']:.6f})")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
vmax = max(c_true.max(), best['c_pred'].max())
levels = np.linspace(0, vmax, 30)

ax = axes[0]
cp = ax.contourf(U_grid, V_grid, c_true, levels=levels, cmap='viridis')
fig.colorbar(cp, ax=ax)
ax.set_title('c(u,v) vera'); ax.set_xlabel('u'); ax.set_ylabel('v')

ax = axes[1]
cp = ax.contourf(U_grid, V_grid, best['c_pred'], levels=levels, cmap='viridis')
fig.colorbar(cp, ax=ax)
ax.set_title(f"ĉ(u,v) — {best['name']}"); ax.set_xlabel('u'); ax.set_ylabel('v')

ax = axes[2]
err = np.abs(c_true - best['c_pred'])
cp = ax.contourf(U_grid, V_grid, err, levels=20, cmap='Reds')
fig.colorbar(cp, ax=ax)
ax.set_title(f"|c - ĉ|  (max={err.max():.2f}, media={err.mean():.4f})")
ax.set_xlabel('u'); ax.set_ylabel('v')

plt.suptitle(f"Ablation best: {best['name']} — KL = {best['kl']:.4f}",
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_current_figure(plots_dir, 'ablation_best_confronto.png', legacy_plots_dir)
plt.close()

# --- 3D per il best ---
step = max(1, GRID_EVAL // 80)
Us, Vs = U_grid[::step, ::step], V_grid[::step, ::step]
Ct, Cp = c_true[::step, ::step], best['c_pred'][::step, ::step]

fig = plt.figure(figsize=(14, 5.5))
fig.suptitle(f"Ablation best: {best['name']} — Superficie 3D",
             fontsize=13, fontweight='bold')
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(Us, Vs, Ct, cmap='viridis', alpha=0.85, edgecolor='none')
ax1.set_xlabel('u'); ax1.set_ylabel('v'); ax1.set_zlabel('c(u,v)')
ax1.set_title('Vera'); ax1.view_init(elev=25, azim=-50)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(Us, Vs, Cp, cmap='viridis', alpha=0.85, edgecolor='none')
ax2.set_xlabel('u'); ax2.set_ylabel('v'); ax2.set_zlabel('ĉ(u,v)')
ax2.set_title('Stimata (ANN)'); ax2.view_init(elev=25, azim=-50)

zmax = max(Ct.max(), Cp.max())
ax1.set_zlim(0, zmax); ax2.set_zlim(0, zmax)
plt.tight_layout()
save_current_figure(plots_dir, 'ablation_best_3d.png', legacy_plots_dir)
plt.close()

best_model_path = save_model_state(
    best['model'],
    run_root,
    legacy_artifacts=WRITE_LEGACY_ARTIFACTS,
)
results_path = save_results(
    run_root,
    {
        'best': {
            'name': best['name'],
            'group': best['group'],
            'kl': float(best['kl']),
            'integral': float(best['integ']),
            'err_max': float(best['err_max']),
            'err_mean': float(best['err_mean']),
            'l1_joint': float(best['l1_joint']),
        },
        'all_configs': [
            {
                'name': r['name'],
                'group': r['group'],
                'hidden': r['hidden'],
                'layers': r['layers'],
                'lam': r['lam'],
                'n': r['N'],
                'transform': bool(r['transform']),
                'epochs': r['epochs'],
                'lr': r['lr'],
                'kl': float(r['kl']),
                'integral': float(r['integ']),
                'err_max': float(r['err_max']),
                'err_mean': float(r['err_mean']),
                'l1_joint': float(r['l1_joint']),
            }
            for r in results
        ],
    },
)

print(f"\nBest model salvato in {best_model_path}")
print(f"Plot salvati in {plots_dir}")
print(f"Risultati salvati in {results_path}")
