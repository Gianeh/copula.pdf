"""
Progetto SML: Stima della Densità di Copula tramite Reti Neurali
================================================================
Team: Edoardo Caproni, Pietro Pianigiani, Tommaso Quintabà

Script demo della pipeline baseline del checkpoint 1 (softplus, nessuna trasformazione).
Artefatti salvati in outputs/<run_id>/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from copula import (
    true_copula_on_grid, generate_data,
    CopulaDensityNet, train, predict_on_grid, kl_divergence
)
from experiment_utils import (
    init_run, save_current_figure, save_model_state, save_results
)

# --- Parametri ---
RHO = 0.7
N = 2000
EPOCHS = 5000
LR = 1e-3
LAMBDA = 10.0
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
        'epochs': EPOCHS,
        'lr': LR,
        'lambda': LAMBDA,
        'grid_eval': GRID_EVAL,
        'output_act': OUTPUT_ACT,
        'input_transform': INPUT_TRANSFORM,
    },
    legacy_artifacts=WRITE_LEGACY_ARTIFACTS,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Run directory: {run_root}\n")

# --- Fase 1-3: dati ---
print("=== Fasi 1-3: Ground truth e pseudo-osservazioni ===")
X, Y, U, V = generate_data(RHO, N, seed=SEED)
print(f"  Generati {N} campioni, ρ = {RHO}")
print(f"  Pseudo-osservazioni: U ∈ [{U.min():.4f}, {U.max():.4f}], "
      f"V ∈ [{V.min():.4f}, {V.max():.4f}]\n")

# --- Fase 4: training ---
print("=== Fase 4: Training ANN ===")
model = CopulaDensityNet(
    hidden=128,
    layers=3,
    output_act=OUTPUT_ACT,
    input_transform=INPUT_TRANSFORM,
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Architettura: 2→128→128→128→1 ({OUTPUT_ACT}), parametri: {n_params}")
print(f"  Loss: NLL + {LAMBDA}·(∫∫ĉ-1)²,  Adam lr={LR},  epoche={EPOCHS}\n")
history = train(model, U, V, device, epochs=EPOCHS, lr=LR, lam=LAMBDA)

# --- Fase 5: valutazione ---
print("\n=== Fase 5: Valutazione ===")
U_grid, V_grid, c_true = true_copula_on_grid(RHO, GRID_EVAL)
_, _, c_pred = predict_on_grid(model, device, GRID_EVAL)

eps = 0.01
du = (1 - 2 * eps) / (GRID_EVAL - 1)
kl = kl_divergence(c_true, c_pred, du)
err = np.abs(c_true - c_pred)
integ = np.trapz(np.trapz(c_pred, dx=du, axis=1), dx=du)
print(f"  KL(c_true ‖ ĉ_pred) = {kl:.6f}")

# --- Plot 1: confronto densità (3 pannelli) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle(f'Copula Gaussiana (ρ={RHO}) — KL = {kl:.4f}',
             fontsize=13, fontweight='bold')

vmax = max(c_true.max(), c_pred.max())
levels = np.linspace(0, vmax, 25)

ax = axes[0]
cp = ax.contourf(U_grid, V_grid, c_true, levels=levels, cmap='viridis')
fig.colorbar(cp, ax=ax)
ax.set_title('c(u,v) vera')
ax.set_xlabel('u')
ax.set_ylabel('v')

ax = axes[1]
cp = ax.contourf(U_grid, V_grid, c_pred, levels=levels, cmap='viridis')
fig.colorbar(cp, ax=ax)
ax.set_title('ĉ(u,v) stimata (ANN)')
ax.set_xlabel('u')
ax.set_ylabel('v')

ax = axes[2]
cp = ax.contourf(U_grid, V_grid, err, levels=20, cmap='Reds')
fig.colorbar(cp, ax=ax)
ax.set_title(f'|c - ĉ|  (max={err.max():.2f}, media={err.mean():.3f})')
ax.set_xlabel('u')
ax.set_ylabel('v')

plt.tight_layout()
plot_path = save_current_figure(plots_dir, 'fase5_confronto.png', legacy_plots_dir)
print(f"  Plot salvato in {plot_path}")
plt.close()

# --- Plot 2: curva di apprendimento ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle('Curva di apprendimento', fontsize=13, fontweight='bold')

ax1.plot(history['nll'], color='steelblue', lw=1.2)
ax1.set_ylabel('NLL (Negative Log-Likelihood)')
ax1.set_title('NLL — più bassa = la rete spiega meglio i dati')
ax1.grid(True, alpha=0.3)

ax2.plot(history['penalty'], color='indianred', lw=1.2)
ax2.set_ylabel(f'λ · (∫∫ĉ − 1)²   [λ={LAMBDA}]')
ax2.set_xlabel('Epoca')
ax2.set_title('Penalty normalizzazione — più bassa = integrale vicino a 1')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
curve_path = save_current_figure(plots_dir, 'fase5_learning_curve.png', legacy_plots_dir)
print(f"  Plot salvato in {curve_path}")
plt.close()

model_path = save_model_state(model, run_root, legacy_artifacts=WRITE_LEGACY_ARTIFACTS, filename='run_model.pt')
print(f"  Modello salvato in {model_path}")

results_path = save_results(
    run_root,
    {
        'metrics': {
            'kl': float(kl),
            'integral': float(integ),
            'err_max': float(err.max()),
            'err_mean': float(err.mean()),
        }
    },
)
print(f"  Risultati salvati in {results_path}")
print("\nDone.")
