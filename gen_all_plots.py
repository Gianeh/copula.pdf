"""
Genera tutti i plot per il report LaTeX:
  00) Dataset triviale: scatter + marginali
  01) Esperimento naive (hidden=64, layers=2, lam=100)
  02) Best config da grid search (hidden=128, layers=3, lam=10)

Per ogni esperimento: contour 2D, superficie 3D, learning curve,
e ricostruzione della PDF congiunta.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import torch

from copula import (
    true_copula_density, true_copula_on_grid, generate_data,
    CopulaDensityNet, train, predict_on_grid, kl_divergence
)
from experiment_utils import (
    init_run, save_current_figure, save_results, set_global_seed
)

RHO = 0.7
N = 2000
GRID_EVAL = 200
SEED = 42
OUTPUT_ACT = 'softplus'
INPUT_TRANSFORM = False
WRITE_LEGACY_ARTIFACTS = os.getenv('WRITE_LEGACY_ARTIFACTS', '0') == '1'

run_root, plots_dir, legacy_plots_dir, _ = init_run(
    __file__,
    seed=SEED,
    config={
        'rho': RHO,
        'n': N,
        'grid_eval': GRID_EVAL,
        'output_act': OUTPUT_ACT,
        'input_transform': INPUT_TRANSFORM,
        'experiments': [
            {'tag': '01_naive', 'hidden': 64, 'layers': 2, 'lam': 100, 'lr': 1e-3, 'epochs': 5000},
            {'tag': '02_best', 'hidden': 128, 'layers': 3, 'lam': 10, 'lr': 1e-3, 'epochs': 5000},
        ],
    },
    legacy_artifacts=WRITE_LEGACY_ARTIFACTS,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Run directory: {run_root}\n")

X, Y, U, V = generate_data(RHO, N, seed=SEED)
U_grid, V_grid, c_true = true_copula_on_grid(RHO, GRID_EVAL)
eps = 0.01
du = (1 - 2 * eps) / (GRID_EVAL - 1)


# =========================================================================
# 00) DATASET TRIVIALE + MARGINALI
# =========================================================================
print("=== 00: Dataset e marginali ===")

fig = plt.figure(figsize=(14, 4.5))

# Scatter congiunta
ax1 = fig.add_subplot(1, 3, 1)
ax1.scatter(X, Y, s=3, alpha=0.3, color='steelblue')
ax1.set_xlabel('X'); ax1.set_ylabel('Y')
ax1.set_title(f'Gaussiana Bivariata (N={N}, ρ={RHO})')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.2)

# Marginale X
ax2 = fig.add_subplot(1, 3, 2)
x_range = np.linspace(-4, 4, 300)
ax2.hist(X, bins=50, density=True, alpha=0.5, color='steelblue', label='Empirica')
ax2.plot(x_range, stats.norm.pdf(x_range), 'k-', lw=1.5, label='N(0,1) teorica')
ax2.set_xlabel('x'); ax2.set_ylabel('Densità')
ax2.set_title('Marginale X')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

# Marginale Y
ax3 = fig.add_subplot(1, 3, 3)
ax3.hist(Y, bins=50, density=True, alpha=0.5, color='indianred', label='Empirica')
ax3.plot(x_range, stats.norm.pdf(x_range), 'k-', lw=1.5, label='N(0,1) teorica')
ax3.set_xlabel('y'); ax3.set_ylabel('Densità')
ax3.set_title('Marginale Y')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2)

plt.tight_layout()
path = save_current_figure(plots_dir, '00_dataset.png', legacy_plots_dir)
plt.close()
print(f"  Salvato {path}")

# Scatter delle pseudo-osservazioni
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.scatter(X, Y, s=3, alpha=0.3, color='steelblue')
ax1.set_xlabel('X'); ax1.set_ylabel('Y')
ax1.set_title('Spazio originale (X, Y)')
ax1.set_aspect('equal'); ax1.grid(True, alpha=0.2)

ax2.scatter(U, V, s=3, alpha=0.3, color='darkorange')
ax2.set_xlabel('u'); ax2.set_ylabel('v')
ax2.set_title('Pseudo-osservazioni (u, v) ∈ [0,1]²')
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.set_aspect('equal'); ax2.grid(True, alpha=0.2)

plt.tight_layout()
path = save_current_figure(plots_dir, '00_pseudo_obs.png', legacy_plots_dir)
plt.close()
print(f"  Salvato {path}")


# =========================================================================
# FUNZIONI COMUNI PER I PLOT
# =========================================================================

def plot_contour(c_true, c_pred, U_grid, V_grid, kl, tag):
    """Plot contour 2D: vera, stimata, errore."""
    vmax = max(c_true.max(), c_pred.max())
    levels = np.linspace(0, vmax, 25)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f'{tag} — KL = {kl:.4f}', fontsize=13, fontweight='bold')

    ax = axes[0]
    cp = ax.contourf(U_grid, V_grid, c_true, levels=levels, cmap='viridis')
    fig.colorbar(cp, ax=ax)
    ax.set_title('c(u,v) vera'); ax.set_xlabel('u'); ax.set_ylabel('v')

    ax = axes[1]
    cp = ax.contourf(U_grid, V_grid, c_pred, levels=levels, cmap='viridis')
    fig.colorbar(cp, ax=ax)
    ax.set_title('ĉ(u,v) stimata (ANN)'); ax.set_xlabel('u'); ax.set_ylabel('v')

    ax = axes[2]
    err = np.abs(c_true - c_pred)
    cp = ax.contourf(U_grid, V_grid, err, levels=20, cmap='Reds')
    fig.colorbar(cp, ax=ax)
    ax.set_title(f'|c - ĉ|  (max={err.max():.2f}, media={err.mean():.3f})')
    ax.set_xlabel('u'); ax.set_ylabel('v')

    plt.tight_layout()
    save_current_figure(plots_dir, f'{tag}_confronto.png', legacy_plots_dir)
    plt.close()


def plot_3d(c_true, c_pred, U_grid, V_grid, tag):
    """Superfici 3D: vera e stimata affiancate."""
    # Subsampling per leggibilità del plot 3D
    step = max(1, GRID_EVAL // 80)
    Us = U_grid[::step, ::step]
    Vs = V_grid[::step, ::step]
    Ct = c_true[::step, ::step]
    Cp = c_pred[::step, ::step]

    fig = plt.figure(figsize=(14, 5.5))
    fig.suptitle(f'{tag} — Superficie 3D', fontsize=13, fontweight='bold')

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(Us, Vs, Ct, cmap='viridis', alpha=0.85, edgecolor='none')
    ax1.set_xlabel('u'); ax1.set_ylabel('v'); ax1.set_zlabel('c(u,v)')
    ax1.set_title('Vera')
    ax1.view_init(elev=25, azim=-50)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(Us, Vs, Cp, cmap='viridis', alpha=0.85, edgecolor='none')
    ax2.set_xlabel('u'); ax2.set_ylabel('v'); ax2.set_zlabel('ĉ(u,v)')
    ax2.set_title('Stimata (ANN)')
    ax2.view_init(elev=25, azim=-50)

    # Z-axis condiviso
    zmax = max(Ct.max(), Cp.max())
    ax1.set_zlim(0, zmax); ax2.set_zlim(0, zmax)

    plt.tight_layout()
    save_current_figure(plots_dir, f'{tag}_3d.png', legacy_plots_dir)
    plt.close()


def plot_learning(history, lam, tag):
    """Curva di apprendimento (NLL + penalty separati)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f'{tag} — Curva di apprendimento', fontsize=13, fontweight='bold')

    ax1.plot(history['nll'], color='steelblue', lw=1.0)
    ax1.set_ylabel('NLL'); ax1.set_title('Negative Log-Likelihood')
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['penalty'], color='indianred', lw=1.0)
    ax2.set_ylabel(f'λ·(∫∫ĉ−1)²  [λ={lam}]'); ax2.set_xlabel('Epoca')
    ax2.set_title('Penalty normalizzazione')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_current_figure(plots_dir, f'{tag}_learning.png', legacy_plots_dir)
    plt.close()


def plot_joint_reconstruction(model, X, Y, RHO, device, tag):
    """
    Ricostruzione della PDF congiunta tramite Sklar:
        f_hat(x,y) = c_hat(F_X(x), F_Y(y)) * f_X(x) * f_Y(y)
    Confrontata con la PDF congiunta vera (gaussiana bivariata).
    """
    grid_size = 150
    x_range = np.linspace(-3.5, 3.5, grid_size)
    y_range = np.linspace(-3.5, 3.5, grid_size)
    Xg, Yg = np.meshgrid(x_range, y_range, indexing='ij')

    # PDF congiunta vera
    cov = [[1, RHO], [RHO, 1]]
    pos = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    f_true = stats.multivariate_normal.pdf(pos, mean=[0, 0], cov=cov).reshape(grid_size, grid_size)

    # Ricostruzione via Sklar: f_hat(x,y) = c_hat(Phi(x), Phi(y)) * phi(x) * phi(y)
    u_vals = stats.norm.cdf(Xg)
    v_vals = stats.norm.cdf(Yg)

    uv_tensor = torch.tensor(
        np.column_stack((u_vals.ravel(), v_vals.ravel())),
        dtype=torch.float32, device=device
    )
    with torch.no_grad():
        c_hat = model(uv_tensor).cpu().numpy().reshape(grid_size, grid_size)

    f_X = stats.norm.pdf(Xg)
    f_Y = stats.norm.pdf(Yg)
    f_reconstructed = c_hat * f_X * f_Y

    # Plot
    vmax = max(f_true.max(), f_reconstructed.max())
    levels = np.linspace(0, vmax, 25)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f'{tag} — Ricostruzione PDF congiunta (Sklar)',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    cp = ax.contourf(Xg, Yg, f_true, levels=levels, cmap='viridis')
    fig.colorbar(cp, ax=ax)
    ax.set_title('f(x,y) vera'); ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = axes[1]
    cp = ax.contourf(Xg, Yg, f_reconstructed, levels=levels, cmap='viridis')
    fig.colorbar(cp, ax=ax)
    ax.set_title('f̂(x,y) = ĉ·φ_X·φ_Y'); ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = axes[2]
    err = np.abs(f_true - f_reconstructed)
    cp = ax.contourf(Xg, Yg, err, levels=20, cmap='Reds')
    fig.colorbar(cp, ax=ax)
    ax.set_title(f'|f - f̂|  (max={err.max():.4f}, media={err.mean():.4f})')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    plt.tight_layout()
    save_current_figure(plots_dir, f'{tag}_joint_pdf.png', legacy_plots_dir)
    plt.close()

    # Calcola errore integrato
    dx = (x_range[-1] - x_range[0]) / (grid_size - 1)
    l1_err = np.trapz(np.trapz(err, dx=dx, axis=1), dx=dx)
    return l1_err


# =========================================================================
# ESPERIMENTI
# =========================================================================

def run_experiment(hidden, layers, lam, lr, epochs, tag, seed_offset):
    print(f"\n=== {tag} ===")
    set_global_seed(SEED + seed_offset)
    model = CopulaDensityNet(
        hidden=hidden,
        layers=layers,
        output_act=OUTPUT_ACT,
        input_transform=INPUT_TRANSFORM,
    ).to(device)
    history = train(model, U, V, device, epochs=epochs, lr=lr, lam=lam)

    _, _, c_pred = predict_on_grid(model, device, GRID_EVAL)
    kl = kl_divergence(c_true, c_pred, du)
    print(f"  KL = {kl:.6f}")

    plot_contour(c_true, c_pred, U_grid, V_grid, kl, tag)
    print(f"  Salvato plots/{tag}_confronto.png")

    plot_3d(c_true, c_pred, U_grid, V_grid, tag)
    print(f"  Salvato plots/{tag}_3d.png")

    plot_learning(history, lam, tag)
    print(f"  Salvato plots/{tag}_learning.png")

    l1 = plot_joint_reconstruction(model, X, Y, RHO, device, tag)
    print(f"  Salvato plots/{tag}_joint_pdf.png")
    print(f"  Errore L1 ricostruzione PDF congiunta = {l1:.6f}")

    return kl, l1


# 01) Naive
kl1, l1_1 = run_experiment(hidden=64, layers=2, lam=100, lr=1e-3, epochs=5000,
                           tag='01_naive', seed_offset=1)

# 02) Best da grid search
kl2, l1_2 = run_experiment(hidden=128, layers=3, lam=10, lr=1e-3, epochs=5000,
                           tag='02_best', seed_offset=2)

print("\n" + "=" * 50)
print("Riepilogo:")
print(f"  Naive:  KL={kl1:.4f}, L1 joint={l1_1:.6f}")
print(f"  Best:   KL={kl2:.4f}, L1 joint={l1_2:.6f}")

results_path = save_results(
    run_root,
    {
        'naive': {'kl': float(kl1), 'l1_joint': float(l1_1)},
        'best': {'kl': float(kl2), 'l1_joint': float(l1_2)},
    },
)
print(f"Risultati salvati in {results_path}")
