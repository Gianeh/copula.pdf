import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# FASE 1-2: GROUND TRUTH ANALITICO
# =============================================================================
#
# Gaussiana Bivariata standard con correlazione rho.
# La densità della copula gaussiana ha forma chiusa (Teorema di Sklar):
#
#   c(u,v) = f(Φ⁻¹(u), Φ⁻¹(v)) / (φ(Φ⁻¹(u)) · φ(Φ⁻¹(v)))
#
# che si semplifica in:
#
#   c(u,v) = (1/√(1-ρ²)) · exp(-(ρ²(x²+y²) - 2ρxy) / (2(1-ρ²)))
#
# dove x = Φ⁻¹(u), y = Φ⁻¹(v) e Φ è la CDF della normale standard.
# =============================================================================

def true_copula_density(u, v, rho):
    """Densità analitica della copula gaussiana bivariata."""
    x = stats.norm.ppf(u)
    y = stats.norm.ppf(v)
    coeff = 1.0 / np.sqrt(1 - rho**2)
    exponent = -(rho**2 * (x**2 + y**2) - 2 * rho * x * y) / (2 * (1 - rho**2))
    return coeff * np.exp(exponent)


def true_copula_on_grid(rho, grid_size=200, eps=0.01):
    """Valuta la densità vera su una griglia uniforme [eps, 1-eps]²."""
    u = np.linspace(eps, 1 - eps, grid_size)
    v = np.linspace(eps, 1 - eps, grid_size)
    U, V = np.meshgrid(u, v, indexing='ij')
    C = true_copula_density(U, V, rho)
    return U, V, C


# =============================================================================
# FASE 3: DATASET E PSEUDO-OSSERVAZIONI
# =============================================================================
#
# Generiamo N campioni (x_i, y_i) dalla congiunta e li trasformiamo in
# pseudo-osservazioni (u_i, v_i) ∈ (0,1)² tramite la trasformazione dei ranghi:
#
#   u_i = rank(x_i) / (N + 1)
#
# Il denominatore N+1 (anziché N) evita valori esattamente 0 o 1,
# prevenendo problemi numerici con log(0) o Φ⁻¹(1) = ∞.
# =============================================================================

def generate_data(rho, N, seed=42):
    """Genera campioni da Gaussiana Bivariata e calcola pseudo-osservazioni."""
    np.random.seed(seed)
    data = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], N)
    X, Y = data[:, 0], data[:, 1]

    U = stats.rankdata(X) / (N + 1)
    V = stats.rankdata(Y) / (N + 1)

    return X, Y, U, V


# =============================================================================
# FASE 4: MODELLO (ANN) E TRAINING
# =============================================================================
#
# Architettura: MLP semplice
#   Input:  (u, v) ∈ [0,1]²
#   Hidden: 2 strati da 64 neuroni, attivazione tanh
#   Output: softplus(z) > 0  (garantisce positività della densità stimata)
#
# Loss function:
#   L(θ) = NLL + λ · penalty
#
#   NLL     = -(1/N) Σ log ĉ_θ(u_i, v_i)       (Maximum Likelihood)
#   penalty = (∫∫ ĉ_θ(u,v) du dv  -  1)²        (vincolo normalizzazione)
#
# L'integrale è approssimato numericamente su una griglia fissa con
# la regola dei trapezi.
# =============================================================================

class CopulaDensityNet(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        net = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*net)

    def forward(self, uv):
        return F.softplus(self.net(uv)).squeeze(-1)


def numerical_integral(model, device, grid_size=50):
    """Approssima ∫∫ ĉ(u,v) du dv su [0.01, 0.99]² con regola dei trapezi."""
    t = torch.linspace(0.01, 0.99, grid_size, device=device)
    uu, vv = torch.meshgrid(t, t, indexing='ij')
    uv = torch.stack([uu.flatten(), vv.flatten()], dim=1)
    c = model(uv).reshape(grid_size, grid_size)
    dx = (0.99 - 0.01) / (grid_size - 1)
    return torch.trapezoid(torch.trapezoid(c, dx=dx, dim=1), dx=dx, dim=0)


def train(model, U, V, device, epochs=3000, lr=1e-3, lam=10.0, grid_size=50,
          print_every=500):
    """
    Addestra la rete minimizzando NLL + λ·(integrale-1)².
    Restituisce lo storico delle loss.
    """
    uv = torch.tensor(np.column_stack((U, V)), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'total': [], 'nll': [], 'penalty': []}

    for epoch in range(1, epochs + 1):
        model.train()
        c_pred = model(uv)
        nll = -torch.log(c_pred + 1e-8).mean()
        integ = numerical_integral(model, device, grid_size)
        pen = (integ - 1.0) ** 2
        loss = nll + lam * pen

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history['total'].append(loss.item())
        history['nll'].append(nll.item())
        history['penalty'].append((lam * pen).item())

        if epoch % print_every == 0 or epoch == 1:
            print(f"  Epoca {epoch:>5d}/{epochs}  |  loss={loss.item():.4f}  "
                  f"nll={nll.item():.4f}  ∫∫ĉ={integ.item():.4f}  "
                  f"pen={pen.item():.6f}")

    return history


# =============================================================================
# FASE 5: VALUTAZIONE
# =============================================================================
#
# Metriche:
#   KL(c_true ‖ ĉ) = ∫∫ c_true(u,v) · log(c_true(u,v) / ĉ(u,v)) du dv
#
# Approssimazione numerica su griglia fine (200×200).
# =============================================================================

def predict_on_grid(model, device, grid_size=200, eps=0.01):
    """Valuta il modello su una griglia uniforme [eps, 1-eps]²."""
    u = np.linspace(eps, 1 - eps, grid_size)
    v = np.linspace(eps, 1 - eps, grid_size)
    U, V = np.meshgrid(u, v, indexing='ij')
    uv = torch.tensor(np.column_stack((U.ravel(), V.ravel())),
                       dtype=torch.float32, device=device)
    with torch.no_grad():
        c_pred = model(uv).cpu().numpy().reshape(grid_size, grid_size)
    return U, V, c_pred


def kl_divergence(c_true, c_pred, du):
    """KL(c_true ‖ c_pred) approssimata su griglia.
    Normalizza entrambe le densità sulla griglia prima del calcolo,
    altrimenti KL può risultare negativa per artefatti numerici."""
    # Normalizzazione sulla griglia
    Z_true = np.trapz(np.trapz(c_true, dx=du, axis=1), dx=du)
    Z_pred = np.trapz(np.trapz(c_pred, dx=du, axis=1), dx=du)
    p = c_true / Z_true
    q = c_pred / Z_pred

    ratio = np.clip(p / (q + 1e-10), 1e-10, None)
    integrand = p * np.log(ratio)
    return np.trapz(np.trapz(integrand, dx=du, axis=1), dx=du)
