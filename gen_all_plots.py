"""
Genera tutti i plot per il report LaTeX:
  1) Esperimento naive iniziale (hidden=64, layers=2, lam=100)
  2) Best config da grid search (hidden=128, layers=3, lam=10)
Salva con timestamp nel nome.
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
N = 2000
GRID_EVAL = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, Y, U, V = generate_data(RHO, N)
U_grid, V_grid, c_true = true_copula_on_grid(RHO, GRID_EVAL)
eps = 0.01
du = (1 - 2 * eps) / (GRID_EVAL - 1)

os.makedirs('plots', exist_ok=True)


def run_and_plot(hidden, layers, lam, lr, epochs, tag):
    print(f"\n=== {tag} ===")
    model = CopulaDensityNet(hidden=hidden, layers=layers).to(device)
    history = train(model, U, V, device, epochs=epochs, lr=lr, lam=lam)

    _, _, c_pred = predict_on_grid(model, device, GRID_EVAL)
    kl = kl_divergence(c_true, c_pred, du)
    print(f"  KL = {kl:.6f}")

    vmax = max(c_true.max(), c_pred.max())
    levels = np.linspace(0, vmax, 25)

    # --- Confronto densità ---
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
    fname = f'plots/{tag}_confronto.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Salvato {fname}")

    # --- Learning curve ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f'{tag} — Curva di apprendimento', fontsize=13, fontweight='bold')

    ax1.plot(history['nll'], color='steelblue', lw=1.0)
    ax1.set_ylabel('NLL')
    ax1.set_title('Negative Log-Likelihood')
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['penalty'], color='indianred', lw=1.0)
    ax2.set_ylabel(f'λ·(∫∫ĉ−1)²  [λ={lam}]')
    ax2.set_xlabel('Epoca')
    ax2.set_title('Penalty normalizzazione')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'plots/{tag}_learning.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Salvato {fname}")

    return kl


# 1) Naive iniziale
run_and_plot(hidden=64, layers=2, lam=100, lr=1e-3, epochs=5000, tag='01_naive')

# 2) Best da grid search
run_and_plot(hidden=128, layers=3, lam=10, lr=1e-3, epochs=5000, tag='02_best')
