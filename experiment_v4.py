"""
Esperimento v4: trasformazione degli input Φ⁻¹(u), Φ⁻¹(v).

Idea: invece di dare (u,v) ∈ [0,1]² alla rete, mappiamo nello spazio
normale (x,y) = (Φ⁻¹(u), Φ⁻¹(v)). In questo spazio la copula gaussiana
è una funzione liscia (simile a una gaussiana riscalata), senza le
divergenze che rendono difficile l'apprendimento ai bordi di [0,1]².
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

configs = [
    # Riferimento: best checkpoint1 (senza trasformazione)
    {'name': 'ref_no_transform',
     'N': 2000, 'act': 'softplus', 'hidden': 128, 'layers': 3,
     'epochs': 5000, 'lr': 1e-3, 'lam': 10, 'grid_int': 50,
     'sched': None, 'clip': None, 'transform': False},

    # Trasformazione + softplus (stessa rete, stessi dati)
    {'name': 'transform_sp_2k',
     'N': 2000, 'act': 'softplus', 'hidden': 128, 'layers': 3,
     'epochs': 5000, 'lr': 1e-3, 'lam': 10, 'grid_int': 50,
     'sched': None, 'clip': None, 'transform': True},

    # Trasformazione + softplus + più dati
    {'name': 'transform_sp_10k',
     'N': 10000, 'act': 'softplus', 'hidden': 128, 'layers': 3,
     'epochs': 10000, 'lr': 1e-3, 'lam': 10, 'grid_int': 100,
     'sched': 'cosine', 'clip': None, 'transform': True},

    # Trasformazione + exp + più dati (ora exp dovrebbe essere stabile)
    {'name': 'transform_exp_10k',
     'N': 10000, 'act': 'exp', 'hidden': 128, 'layers': 3,
     'epochs': 10000, 'lr': 1e-3, 'lam': 10, 'grid_int': 100,
     'sched': 'cosine', 'clip': 1.0, 'transform': True},
]

results = []

for cfg in configs:
    name = cfg['name']
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  N={cfg['N']}, {cfg['act']}, {cfg['hidden']}x{cfg['layers']}, "
          f"transform={cfg['transform']}, clip={cfg['clip']}, sched={cfg['sched']}")
    print(f"{'='*60}")

    X, Y, U, V = generate_data(RHO, cfg['N'], seed=SEED)
    set_global_seed(SEED + len(results) + 1)
    model = CopulaDensityNet(hidden=cfg['hidden'], layers=cfg['layers'],
                             output_act=cfg['act'],
                             input_transform=cfg['transform']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametri: {n_params}")

    history = train(model, U, V, device,
                    epochs=cfg['epochs'], lr=cfg['lr'], lam=cfg['lam'],
                    grid_size=cfg['grid_int'], print_every=2000,
                    scheduler_type=cfg['sched'], grad_clip=cfg['clip'])

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
                    'l1_joint': l1_joint,
                    'c_pred': c_pred, 'model': model, 'history': history})

    print(f"\n  KL={kl:.6f}  ∫∫ĉ={integ:.4f}  err_max={err.max():.3f}  "
          f"err_mean={err.mean():.4f}  L1_joint={l1_joint:.4f}")

# --- Tabella ---
print(f"\n{'='*100}")
print(f"{'Config':<22} {'Act':<9} {'N':>6} {'Tr':>3} {'clip':>5} "
      f"{'KL':>10} {'∫∫ĉ':>7} {'err_max':>8} {'err_mean':>9} {'L1_joint':>9}")
print("-" * 100)
for r in results:
    hl = f"{r['hidden']}x{r['layers']}"
    tr = 'Y' if r['transform'] else 'N'
    print(f"{r['name']:<22} {r['act']:<9} {r['N']:>6} {tr:>3} {str(r['clip']):>5} "
          f"{r['kl']:>10.6f} {r['integ']:>7.4f} {r['err_max']:>8.3f} "
          f"{r['err_mean']:>9.4f} {r['l1_joint']:>9.4f}")

best = min(results, key=lambda r: r['kl'])
print(f"\nBest: {best['name']} (KL={best['kl']:.6f})")

# --- Plot best: contour ---
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

plt.suptitle(f"v4 Best: {best['name']} — KL = {best['kl']:.4f}",
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_current_figure(plots_dir, 'v4_best_confronto.png', legacy_plots_dir)
plt.close()

# --- Plot best: 3D ---
step = max(1, GRID_EVAL // 80)
Us, Vs = U_grid[::step, ::step], V_grid[::step, ::step]
Ct, Cp = c_true[::step, ::step], best['c_pred'][::step, ::step]

fig = plt.figure(figsize=(14, 5.5))
fig.suptitle(f"v4 Best: {best['name']} — Superficie 3D",
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
save_current_figure(plots_dir, 'v4_best_3d.png', legacy_plots_dir)
plt.close()

# --- Plot best: ricostruzione PDF congiunta ---
# Ricomputa per il best
u_vals_b = stats.norm.cdf(Xg)
v_vals_b = stats.norm.cdf(Yg)
uv_tb = torch.tensor(np.column_stack((u_vals_b.ravel(), v_vals_b.ravel())),
                      dtype=torch.float32, device=device)
with torch.no_grad():
    c_hat_b = best['model'](uv_tb).cpu().numpy().reshape(grid_j, grid_j)
f_recon_b = c_hat_b * stats.norm.pdf(Xg) * stats.norm.pdf(Yg)
err_j = np.abs(f_true - f_recon_b)

vmax_j = max(f_true.max(), f_recon_b.max())
levels_j = np.linspace(0, vmax_j, 25)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle(f"v4 Best — Ricostruzione PDF congiunta (Sklar)",
             fontsize=13, fontweight='bold')

ax = axes[0]
cp = ax.contourf(Xg, Yg, f_true, levels=levels_j, cmap='viridis')
fig.colorbar(cp, ax=ax)
ax.set_title('f(x,y) vera'); ax.set_xlabel('x'); ax.set_ylabel('y')

ax = axes[1]
cp = ax.contourf(Xg, Yg, f_recon_b, levels=levels_j, cmap='viridis')
fig.colorbar(cp, ax=ax)
ax.set_title('f̂(x,y) ricostruita'); ax.set_xlabel('x'); ax.set_ylabel('y')

ax = axes[2]
cp = ax.contourf(Xg, Yg, err_j, levels=20, cmap='Reds')
fig.colorbar(cp, ax=ax)
ax.set_title(f'|f - f̂|  (max={err_j.max():.4f}, media={err_j.mean():.4f})')
ax.set_xlabel('x'); ax.set_ylabel('y')

plt.tight_layout()
save_current_figure(plots_dir, 'v4_best_joint.png', legacy_plots_dir)
plt.close()

# --- Plot best: learning curve ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle(f"v4 Best: {best['name']} — Curva di apprendimento",
             fontsize=13, fontweight='bold')

ax1.plot(best['history']['nll'], color='steelblue', lw=1.0)
ax1.set_ylabel('NLL'); ax1.set_title('Negative Log-Likelihood')
ax1.grid(True, alpha=0.3)

ax2.plot(best['history']['penalty'], color='indianred', lw=1.0)
ax2.set_ylabel(f'λ·(∫∫ĉ−1)²  [λ={best["lam"]}]'); ax2.set_xlabel('Epoca')
ax2.set_title('Penalty normalizzazione')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_current_figure(plots_dir, 'v4_best_learning.png', legacy_plots_dir)
plt.close()

# --- Bar chart confronto ---
fig, ax = plt.subplots(figsize=(10, 4.5))
names = [r['name'] for r in results]
kls = [r['kl'] for r in results]
colors = ['#aaaaaa' if not r['transform'] else
          ('#5cb85c' if r['act'] == 'exp' else '#337ab7') for r in results]
bars = ax.bar(names, kls, color=colors)
ax.set_ylabel('KL Divergence')
ax.set_title('Confronto configurazioni v4 — Trasformazione input Φ⁻¹')
ax.grid(True, alpha=0.3, axis='y')
for bar, kl_val in zip(bars, kls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
            f'{kl_val:.4f}', ha='center', fontsize=9)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
save_current_figure(plots_dir, 'v4_comparison.png', legacy_plots_dir)
plt.close()

# Salva best model
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
            'kl': float(best['kl']),
            'integral': float(best['integ']),
            'err_max': float(best['err_max']),
            'err_mean': float(best['err_mean']),
            'l1_joint': float(best['l1_joint']),
        },
        'all_configs': [
            {
                'name': r['name'],
                'act': r['act'],
                'n': r['N'],
                'transform': bool(r['transform']),
                'scheduler': r['sched'],
                'clip': r['clip'],
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
