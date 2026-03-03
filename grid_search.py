"""
Grid search sugli iperparametri della rete per la copula gaussiana (ρ=0.7).
Stampa una tabella ordinata per KL divergence e salva il best model.
"""

import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt

from copula import (
    true_copula_on_grid, generate_data,
    CopulaDensityNet, train, predict_on_grid, kl_divergence
)

# --- Dati (fissi per tutti gli esperimenti) ---
RHO = 0.7
N = 2000
GRID_EVAL = 200
EPOCHS = 5000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

X, Y, U, V = generate_data(RHO, N)
U_grid, V_grid, c_true = true_copula_on_grid(RHO, GRID_EVAL)
eps = 0.01
du = (1 - 2 * eps) / (GRID_EVAL - 1)

# --- Griglia iperparametri ---
param_grid = {
    'hidden':  [32, 64, 128],
    'layers':  [2, 3],
    'lam':     [10, 100],
    'lr':      [1e-3, 5e-4],
}

combos = list(itertools.product(
    param_grid['hidden'], param_grid['layers'],
    param_grid['lam'], param_grid['lr']
))

print(f"Combinazioni da testare: {len(combos)}\n")
print(f"{'#':>3}  {'hidden':>6}  {'layers':>6}  {'lam':>5}  {'lr':>8}  {'KL':>10}  {'integ':>6}")
print("-" * 60)

results = []

for i, (hidden, layers, lam, lr) in enumerate(combos, 1):
    model = CopulaDensityNet(hidden=hidden, layers=layers).to(device)
    history = train(model, U, V, device, epochs=EPOCHS, lr=lr, lam=lam,
                    print_every=99999)  # silenzioso

    _, _, c_pred = predict_on_grid(model, device, GRID_EVAL)
    kl = kl_divergence(c_true, c_pred, du)
    integ = np.trapz(np.trapz(c_pred, dx=du, axis=1), dx=du)

    results.append({
        'hidden': hidden, 'layers': layers, 'lam': lam, 'lr': lr,
        'kl': kl, 'integ': integ, 'history': history, 'model': model
    })

    print(f"{i:>3}  {hidden:>6}  {layers:>6}  {lam:>5}  {lr:>8.0e}  {kl:>10.6f}  {integ:>6.3f}")

# --- Classifica ---
results.sort(key=lambda r: r['kl'])

print("\n" + "=" * 60)
print("TOP 5 configurazioni (per KL crescente):")
print(f"{'#':>3}  {'hidden':>6}  {'layers':>6}  {'lam':>5}  {'lr':>8}  {'KL':>10}  {'integ':>6}")
print("-" * 60)
for i, r in enumerate(results[:5], 1):
    print(f"{i:>3}  {r['hidden']:>6}  {r['layers']:>6}  {r['lam']:>5}  {r['lr']:>8.0e}"
          f"  {r['kl']:>10.6f}  {r['integ']:>6.3f}")

# --- Salva best model ---
best = results[0]
torch.save(best['model'].state_dict(), 'best_model.pt')
print(f"\nBest model salvato in best_model.pt")
print(f"  hidden={best['hidden']}, layers={best['layers']}, "
      f"lam={best['lam']}, lr={best['lr']}")
print(f"  KL = {best['kl']:.6f}, integ = {best['integ']:.4f}")
