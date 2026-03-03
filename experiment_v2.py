"""
Esperimenti v2: verso una stima quasi-perfetta della copula gaussiana.
Confronto sistematico su 3 assi di miglioramento:
  1. Output activation: softplus vs exp
  2. Dati: N=2000 vs N=10000
  3. LR scheduling: nessuno vs cosine annealing

Rete fissa: 128x3 (best da grid search).
Epoche: 10000.  Griglia integrale: 100x100.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from copula import (
    true_copula_on_grid, generate_data,
    CopulaDensityNet, train, predict_on_grid, kl_divergence
)

RHO = 0.7
GRID_EVAL = 200
EPOCHS = 10000
HIDDEN = 128
LAYERS = 3
LAM = 10.0
LR = 1e-3
GRID_INT = 100  # griglia integrale più fitta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Ground truth
U_grid, V_grid, c_true = true_copula_on_grid(RHO, GRID_EVAL)
eps = 0.01
du = (1 - 2 * eps) / (GRID_EVAL - 1)

os.makedirs('plots', exist_ok=True)

# Configurazioni da testare
configs = [
    {'name': 'A_softplus_2k',       'N': 2000,  'act': 'softplus', 'sched': None},
    {'name': 'B_exp_2k',            'N': 2000,  'act': 'exp',      'sched': None},
    {'name': 'C_exp_10k',           'N': 10000, 'act': 'exp',      'sched': None},
    {'name': 'D_exp_10k_cosine',    'N': 10000, 'act': 'exp',      'sched': 'cosine'},
]

results = []

for cfg in configs:
    name = cfg['name']
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  N={cfg['N']}, act={cfg['act']}, sched={cfg['sched']}")
    print(f"{'='*60}")

    X, Y, U, V = generate_data(RHO, cfg['N'])

    model = CopulaDensityNet(hidden=HIDDEN, layers=LAYERS,
                             output_act=cfg['act']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametri: {n_params}")

    history = train(model, U, V, device, epochs=EPOCHS, lr=LR, lam=LAM,
                    grid_size=GRID_INT, print_every=2000,
                    scheduler_type=cfg['sched'])

    _, _, c_pred = predict_on_grid(model, device, GRID_EVAL)
    kl = kl_divergence(c_true, c_pred, du)
    integ = np.trapz(np.trapz(c_pred, dx=du, axis=1), dx=du)
    err_max = np.abs(c_true - c_pred).max()
    err_mean = np.abs(c_true - c_pred).mean()

    results.append({**cfg, 'kl': kl, 'integ': integ,
                    'err_max': err_max, 'err_mean': err_mean,
                    'history': history, 'c_pred': c_pred})

    print(f"\n  KL={kl:.6f}  ∫∫ĉ={integ:.4f}  err_max={err_max:.3f}  err_mean={err_mean:.4f}")

# --- Tabella riepilogativa ---
print(f"\n{'='*80}")
print(f"{'Config':<25} {'Act':<10} {'N':>6} {'Sched':<8} {'KL':>10} {'∫∫ĉ':>7} {'err_max':>8} {'err_mean':>9}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<25} {r['act']:<10} {r['N']:>6} {str(r['sched']):<8} "
          f"{r['kl']:>10.6f} {r['integ']:>7.4f} {r['err_max']:>8.3f} {r['err_mean']:>9.4f}")

# --- Plot confronto ---
best = min(results, key=lambda r: r['kl'])
print(f"\nBest: {best['name']} (KL={best['kl']:.6f})")

# Plot confronto densità per il best
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

plt.suptitle(f"Best: {best['name']} — KL = {best['kl']:.4f}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/v2_best_confronto.png', dpi=150)
plt.close()

# 3D del best
from mpl_toolkits.mplot3d import Axes3D
step = max(1, GRID_EVAL // 80)
Us, Vs = U_grid[::step, ::step], V_grid[::step, ::step]
Ct, Cp = c_true[::step, ::step], best['c_pred'][::step, ::step]

fig = plt.figure(figsize=(14, 5.5))
fig.suptitle(f"Best: {best['name']} — Superficie 3D", fontsize=13, fontweight='bold')
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
plt.savefig('plots/v2_best_3d.png', dpi=150)
plt.close()

# Confronto KL tutte le configurazioni (bar chart)
fig, ax = plt.subplots(figsize=(8, 4))
names = [r['name'] for r in results]
kls = [r['kl'] for r in results]
colors = ['#d9534f' if r['act'] == 'softplus' else '#5cb85c' for r in results]
bars = ax.bar(names, kls, color=colors)
ax.set_ylabel('KL Divergence')
ax.set_title('Confronto configurazioni v2')
ax.grid(True, alpha=0.3, axis='y')
for bar, kl in zip(bars, kls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{kl:.4f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('plots/v2_comparison.png', dpi=150)
plt.close()

print("\nPlot salvati in plots/v2_*.png")
