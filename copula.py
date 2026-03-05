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
# Architettura: MLP
#   Input:  (u, v) ∈ [0,1]²
#   Hidden: N strati da H neuroni, attivazione tanh
#   Output: exp(z) o softplus(z) > 0  (garantisce positività)
#
# exp vs softplus:
#   softplus(z) ≈ z per z >> 0, cresce linearmente → satura sui picchi
#   exp(z) cresce esponenzialmente → può raggiungere valori arbitrari
#   Per copule con picchi estremi (es. angoli della gaussiana), exp è meglio.
#
# Loss function:
#   L(θ) = NLL + λ · penalty
#
#   NLL     = -(1/N) Σ log ĉ_θ(u_i, v_i)       (Maximum Likelihood)
#   penalty = (∫∫ ĉ_θ(u,v) du dv  -  1)²        (vincolo normalizzazione)
#
# Con output exp, NLL si semplifica: NLL = -(1/N) Σ z_i  (dove z_i è il
# logit pre-attivazione), evitando log(exp(z)) = z numericamente più stabile.
#
# L'integrale è approssimato numericamente su una griglia fissa con
# la regola dei trapezi.
# =============================================================================

class CopulaDensityNet(nn.Module):
    def __init__(self, hidden=64, layers=2, output_act='softplus', input_transform=False):
        super().__init__()
        self.output_act = output_act
        self.input_transform = input_transform
        net = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*net)

    def _transform(self, uv):
        """Mappa (u,v) ∈ [0,1]² → (Φ⁻¹(u), Φ⁻¹(v)) ∈ ℝ².
        In questo spazio la copula è una funzione liscia, senza divergenze."""
        if not self.input_transform:
            return uv
        uv = torch.clamp(uv, 1e-4, 1 - 1e-4)
        return torch.erfinv(2 * uv - 1) * 1.4142135623730951  # √2

    def forward(self, uv):
        z = self.net(self._transform(uv)).squeeze(-1)
        if self.output_act == 'exp':
            return torch.exp(z)
        else:
            return F.softplus(z)

    def forward_log(self, uv):
        """Restituisce log ĉ(u,v) direttamente, numericamente stabile per NLL."""
        z = self.net(self._transform(uv)).squeeze(-1)
        if self.output_act == 'exp':
            return z  # log(exp(z)) = z
        else:
            return torch.log(F.softplus(z) + 1e-8)


def numerical_integral(model, device, grid_size=50):
    """Approssima ∫∫ ĉ(u,v) du dv su [0.01, 0.99]² con regola dei trapezi."""
    t = torch.linspace(0.01, 0.99, grid_size, device=device)
    uu, vv = torch.meshgrid(t, t, indexing='ij')
    uv = torch.stack([uu.flatten(), vv.flatten()], dim=1)
    c = model(uv).reshape(grid_size, grid_size)
    dx = (0.99 - 0.01) / (grid_size - 1)
    return torch.trapezoid(torch.trapezoid(c, dx=dx, dim=1), dx=dx, dim=0)


def train(model, U, V, device, epochs=3000, lr=1e-3, lam=10.0, grid_size=50,
          print_every=500, scheduler_type=None, grad_clip=None):
    """
    Addestra la rete minimizzando NLL + λ·(integrale-1)².
    Usa forward_log per stabilità numerica con output exp.

    scheduler_type: None, 'cosine', o 'plateau'
    """
    uv = torch.tensor(np.column_stack((U, V)), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=300, factor=0.5, min_lr=1e-6)
    else:
        scheduler = None

    history = {'total': [], 'nll': [], 'penalty': []}

    for epoch in range(1, epochs + 1):
        model.train()
        log_c = model.forward_log(uv)
        nll = -log_c.mean()
        integ = numerical_integral(model, device, grid_size)
        pen = (integ - 1.0) ** 2
        loss = nll + lam * pen

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        elif scheduler is not None:
            scheduler.step()

        history['total'].append(loss.item())
        history['nll'].append(nll.item())
        history['penalty'].append((lam * pen).item())

        if epoch % print_every == 0 or epoch == 1:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoca {epoch:>5d}/{epochs}  |  loss={loss.item():.4f}  "
                  f"nll={nll.item():.4f}  ∫∫ĉ={integ.item():.4f}  "
                  f"pen={pen.item():.6f}  lr={cur_lr:.1e}")

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
