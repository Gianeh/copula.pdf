"""
Progetto SML: Stima della Densità di Copula tramite Reti Neurali
================================================================
Team: Edoardo Caproni, Pietro Pianigiani, Tommaso Quintabà

Questo script esegue l'intera pipeline:
  1-2. Ground truth analitico (copula gaussiana con ρ=0.7)
    3. Generazione dati + pseudo-osservazioni
    4. Training ANN (NLL + penalty normalizzazione)
    5. Valutazione (KL divergence) e visualizzazione
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from copula import (
    true_copula_on_grid, generate_data,
    CopulaDensityNet, train, predict_on_grid, kl_divergence
)

# --- Parametri ---
RHO = 0.7
N = 2000
EPOCHS = 5000
LR = 1e-3
LAMBDA = 100.0
GRID_EVAL = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# --- Fase 1-3: dati ---
print("=== Fasi 1-3: Ground truth e pseudo-osservazioni ===")
X, Y, U, V = generate_data(RHO, N)
print(f"  Generati {N} campioni, ρ = {RHO}")
print(f"  Pseudo-osservazioni: U ∈ [{U.min():.4f}, {U.max():.4f}], "
      f"V ∈ [{V.min():.4f}, {V.max():.4f}]\n")

# --- Fase 4: training ---
print("=== Fase 4: Training ANN ===")
model = CopulaDensityNet(hidden=64, layers=2).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Architettura: 2→64→64→1 (softplus), parametri: {n_params}")
print(f"  Loss: NLL + {LAMBDA}·(∫∫ĉ-1)²,  Adam lr={LR},  epoche={EPOCHS}\n")
history = train(model, U, V, device, epochs=EPOCHS, lr=LR, lam=LAMBDA)

# --- Fase 5: valutazione ---
print("\n=== Fase 5: Valutazione ===")
U_grid, V_grid, c_true = true_copula_on_grid(RHO, GRID_EVAL)
_, _, c_pred = predict_on_grid(model, device, GRID_EVAL)

eps = 0.01
du = (1 - 2 * eps) / (GRID_EVAL - 1)
kl = kl_divergence(c_true, c_pred, du)
print(f"  KL(c_true ‖ ĉ_pred) = {kl:.6f}")

# --- Plot ---
os.makedirs('plots', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(f'Copula Gaussiana (ρ={RHO}) — KL Divergence = {kl:.4f}',
             fontsize=13, fontweight='bold')

# Livelli di contorno condivisi per confronto visivo
vmax = max(c_true.max(), c_pred.max())
levels = np.linspace(0, vmax, 25)

# Densità vera
ax = axes[0, 0]
cp = ax.contourf(U_grid, V_grid, c_true, levels=levels, cmap='viridis')
fig.colorbar(cp, ax=ax)
ax.set_title('c(u,v) vera (analitica)')
ax.set_xlabel('u'); ax.set_ylabel('v')

# Densità stimata
ax = axes[0, 1]
cp = ax.contourf(U_grid, V_grid, c_pred, levels=levels, cmap='viridis')
fig.colorbar(cp, ax=ax)
ax.set_title('ĉ(u,v) stimata (ANN)')
ax.set_xlabel('u'); ax.set_ylabel('v')

# Errore assoluto
ax = axes[1, 0]
err = np.abs(c_true - c_pred)
cp = ax.contourf(U_grid, V_grid, err, levels=20, cmap='Reds')
fig.colorbar(cp, ax=ax)
ax.set_title(f'|c - ĉ|  (max={err.max():.3f}, media={err.mean():.3f})')
ax.set_xlabel('u'); ax.set_ylabel('v')

# Loss curve
ax = axes[1, 1]
ax.plot(history['total'], label='Totale', color='black')
ax.plot(history['nll'], label='NLL', color='blue', ls='--', alpha=0.7)
ax.plot(history['penalty'], label=f'λ·penalty', color='red', ls='--', alpha=0.7)
ax.set_xlabel('Epoca'); ax.set_ylabel('Loss')
ax.set_title('Curva di apprendimento')
ax.legend()
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('plots/fase5_confronto.png', dpi=150)
print(f"  Plot salvato in plots/fase5_confronto.png")
plt.close()

print("\nDone.")
