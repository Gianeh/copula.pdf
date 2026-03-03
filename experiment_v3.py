"""
Esperimento v3: push per risultati quasi-perfetti.
exp + gradient clipping + N grande + cosine LR.
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

RHO = 0.7
GRID_EVAL = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

U_grid, V_grid, c_true = true_copula_on_grid(RHO, GRID_EVAL)
eps = 0.01
du = (1 - 2 * eps) / (GRID_EVAL - 1)

os.makedirs('plots', exist_ok=True)

configs = [
    # Baseline: il nostro best attuale per riferimento
    {'name': 'baseline_softplus',
     'N': 2000, 'act': 'softplus', 'hidden': 128, 'layers': 3,
     'epochs': 10000, 'lr': 1e-3, 'lam': 10, 'grid_int': 50,
     'sched': None, 'clip': None},

    # exp + grad clip + 10k dati
    {'name': 'exp_clip_10k',
     'N': 10000, 'act': 'exp', 'hidden': 128, 'layers': 3,
     'epochs': 10000, 'lr': 1e-3, 'lam': 10, 'grid_int': 100,
     'sched': 'cosine', 'clip': 1.0},

    # exp + grad clip + 50k dati (sfrutta la GPU)
    {'name': 'exp_clip_50k',
     'N': 50000, 'act': 'exp', 'hidden': 128, 'layers': 3,
     'epochs': 10000, 'lr': 1e-3, 'lam': 10, 'grid_int': 100,
     'sched': 'cosine', 'clip': 1.0},

    # Rete più larga + exp + 50k
    {'name': 'exp_wide_50k',
     'N': 50000, 'act': 'exp', 'hidden': 256, 'layers': 4,
     'epochs': 10000, 'lr': 1e-3, 'lam': 10, 'grid_int': 100,
     'sched': 'cosine', 'clip': 1.0},
]

results = []

for cfg in configs:
    name = cfg['name']
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  N={cfg['N']}, {cfg['act']}, {cfg['hidden']}x{cfg['layers']}, "
          f"clip={cfg['clip']}, sched={cfg['sched']}")
    print(f"{'='*60}")

    X, Y, U, V = generate_data(RHO, cfg['N'])
    model = CopulaDensityNet(hidden=cfg['hidden'], layers=cfg['layers'],
                             output_act=cfg['act']).to(device)
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

    results.append({**cfg, 'kl': kl, 'integ': integ,
                    'err_max': err.max(), 'err_mean': err.mean(),
                    'c_pred': c_pred, 'model': model})

    print(f"\n  KL={kl:.6f}  ∫∫ĉ={integ:.4f}  err_max={err.max():.3f}  err_mean={err.mean():.4f}")

# --- Tabella ---
print(f"\n{'='*90}")
print(f"{'Config':<22} {'Act':<9} {'N':>6} {'H×L':>5} {'clip':>5} {'KL':>10} {'∫∫ĉ':>7} {'err_max':>8} {'err_mean':>9}")
print("-" * 90)
for r in results:
    hl = f"{r['hidden']}x{r['layers']}"
    print(f"{r['name']:<22} {r['act']:<9} {r['N']:>6} {hl:>5} {str(r['clip']):>5} "
          f"{r['kl']:>10.6f} {r['integ']:>7.4f} {r['err_max']:>8.3f} {r['err_mean']:>9.4f}")

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

plt.suptitle(f"v3 Best: {best['name']} — KL = {best['kl']:.4f}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/v3_best_confronto.png', dpi=150)
plt.close()

# --- Plot best: 3D ---
step = max(1, GRID_EVAL // 80)
Us, Vs = U_grid[::step, ::step], V_grid[::step, ::step]
Ct, Cp = c_true[::step, ::step], best['c_pred'][::step, ::step]

fig = plt.figure(figsize=(14, 5.5))
fig.suptitle(f"v3 Best: {best['name']} — Superficie 3D", fontsize=13, fontweight='bold')
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
plt.savefig('plots/v3_best_3d.png', dpi=150)
plt.close()

# --- Plot best: ricostruzione PDF congiunta ---
grid_size = 150
x_range = np.linspace(-3.5, 3.5, grid_size)
y_range = np.linspace(-3.5, 3.5, grid_size)
Xg, Yg = np.meshgrid(x_range, y_range, indexing='ij')

cov = [[1, RHO], [RHO, 1]]
pos = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
f_true = stats.multivariate_normal.pdf(pos, mean=[0, 0], cov=cov).reshape(grid_size, grid_size)

u_vals = stats.norm.cdf(Xg)
v_vals = stats.norm.cdf(Yg)
uv_tensor = torch.tensor(np.column_stack((u_vals.ravel(), v_vals.ravel())),
                          dtype=torch.float32, device=device)
with torch.no_grad():
    c_hat = best['model'](uv_tensor).cpu().numpy().reshape(grid_size, grid_size)

f_reconstructed = c_hat * stats.norm.pdf(Xg) * stats.norm.pdf(Yg)

vmax_j = max(f_true.max(), f_reconstructed.max())
levels_j = np.linspace(0, vmax_j, 25)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle(f"v3 Best — Ricostruzione PDF congiunta (Sklar)", fontsize=13, fontweight='bold')

ax = axes[0]
cp = ax.contourf(Xg, Yg, f_true, levels=levels_j, cmap='viridis')
fig.colorbar(cp, ax=ax)
ax.set_title('f(x,y) vera'); ax.set_xlabel('x'); ax.set_ylabel('y')

ax = axes[1]
cp = ax.contourf(Xg, Yg, f_reconstructed, levels=levels_j, cmap='viridis')
fig.colorbar(cp, ax=ax)
ax.set_title('f̂(x,y) ricostruita'); ax.set_xlabel('x'); ax.set_ylabel('y')

ax = axes[2]
err_j = np.abs(f_true - f_reconstructed)
cp = ax.contourf(Xg, Yg, err_j, levels=20, cmap='Reds')
fig.colorbar(cp, ax=ax)
ax.set_title(f'|f - f̂|  (max={err_j.max():.4f}, media={err_j.mean():.4f})')
ax.set_xlabel('x'); ax.set_ylabel('y')

plt.tight_layout()
plt.savefig('plots/v3_best_joint.png', dpi=150)
plt.close()

dx_j = (x_range[-1] - x_range[0]) / (grid_size - 1)
l1_joint = np.trapz(np.trapz(err_j, dx=dx_j, axis=1), dx=dx_j)
print(f"  L1 ricostruzione PDF congiunta = {l1_joint:.6f}")

# --- Bar chart ---
fig, ax = plt.subplots(figsize=(9, 4))
names = [r['name'] for r in results]
kls = [r['kl'] for r in results]
colors = ['#d9534f' if r['act'] == 'softplus' else '#5cb85c' for r in results]
bars = ax.bar(names, kls, color=colors)
ax.set_ylabel('KL Divergence')
ax.set_title('Confronto configurazioni v3')
ax.grid(True, alpha=0.3, axis='y')
for bar, kl_val in zip(bars, kls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
            f'{kl_val:.4f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('plots/v3_comparison.png', dpi=150)
plt.close()

# Salva best model
torch.save(best['model'].state_dict(), 'best_model.pt')
print(f"\nBest model salvato in best_model.pt")
print("Plot salvati in plots/v3_*.png")
