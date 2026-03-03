# Roadmap — Stima della Densità di Copula tramite ANN

**Team:** Edoardo Caproni, Pietro Pianigiani, Tommaso Quintabà
**Corso:** Statistical Machine Learning

---

## Fase A: Caso triviale — Copula Gaussiana Bivariata (ρ = 0.7)

Obiettivo: validare la pipeline su un caso con soluzione analitica nota, fino a ottenere
una stima quasi perfetta che funga da baseline per i casi complessi.

### A.1 — Setup e approccio naive ✅

**Documentato in:** `report/checkpoint1.tex`, Sezioni 1–3

- Ground truth analitico: forma chiusa della copula gaussiana
- Dataset: N=2000 campioni, pseudo-osservazioni via rank transform
- Modello: MLP 2→64→64→1, tanh + softplus
- Loss: NLL + λ·(∫∫ĉ−1)², con λ=100
- Risultato: **KL = 0.026**, err_max = 10.32, L1_joint = 0.169
- Problema: λ=100 troppo forte, plateau nella NLL per ~2000 epoche

### A.2 — Grid Search ✅

**Documentato in:** `report/checkpoint1.tex`, Sezione 4
**Script:** `grid_search.py`

- 24 combinazioni: hidden ∈ {32, 64, 128}, layers ∈ {2, 3}, λ ∈ {10, 100}, lr ∈ {1e-3, 5e-4}
- **Risultato chiave:** λ=10 domina sistematicamente λ=100
- Best: hidden=128, layers=3, λ=10, lr=1e-3 → **KL = 0.007**

### A.3 — Best configuration (post grid search) ✅

**Documentato in:** `report/checkpoint1.tex`, Sezioni 5–6
**Script:** `gen_all_plots.py`

- MLP 2→128→128→128→1, softplus, λ=10, 5000 epoche
- Risultato: **KL = 0.010**, err_max = 4.32, L1_joint = 0.141
- Aggiunta ricostruzione PDF congiunta via Sklar: f̂(x,y) = ĉ(Φ(x),Φ(y))·φ(x)·φ(y)
- La ricostruzione è ottima (err_medio = 0.003) ma i picchi della copula restano sottostimati

### A.4 — Esperimenti v2: attivazione, dati, scheduling ✅

**Documentato in:** `report/checkpoint2.tex`, Sezione 2
**Script:** `experiment_v2.py`

Confronto sistematico su 3 assi, rete fissa 128×3, 10000 epoche:

| Config | Act | N | Scheduler | KL | err_max |
|---|---|---|---|---|---|
| A_softplus_2k | softplus | 2k | — | 0.035 | 8.59 |
| B_exp_2k | exp | 2k | — | 0.017 | 70.78 |
| C_exp_10k | exp | 10k | — | 0.041 | 10.82 |
| D_exp_10k_cosine | exp | 10k | cosine | **0.013** | 16.97 |

**Lezione appresa:** exp riduce la KL ma esplode ai bordi (err_max 10×).
Cosine annealing è essenziale per stabilizzare exp. Senza scheduler,
exp con 10k dati (C) è peggiore di softplus con 2k (A).

### A.5 — Esperimenti v3: gradient clipping + GPU ✅

**Documentato in:** `report/checkpoint2.tex`, Sezione 3
**Script:** `experiment_v3.py`

Push per "quasi perfetto" con GPU, più dati, gradient clipping:

| Config | Act | N | clip | KL | err_max |
|---|---|---|---|---|---|
| baseline_softplus | softplus | 2k | — | 0.033 | 9.80 |
| exp_clip_10k | exp | 10k | 1.0 | **0.011** | 132.6 |
| exp_clip_50k | exp | 50k | 1.0 | **0.011** | 114.2 |
| exp_wide_50k | exp | 50k | 1.0 | 0.013 | 98.5 |

**Lezione appresa:** gradient clipping + cosine stabilizzano parzialmente exp.
KL scende a 0.011, ma err_max resta >100. Rete 256×4 non migliora (più
parametri = più instabilità). Il problema è strutturale, non risolvibile
con trucchi di ottimizzazione.

**Analisi del problema:** la copula gaussiana diverge agli angoli (0,0) e (1,1)
del quadrato unitario, dove Φ⁻¹(u) → ±∞. softplus non riesce a raggiungere
i valori alti; exp li supera e esplode. Servono approcci diversi.

### A.6 — Esperimenti v4: trasformazione degli input ✅

**Documentato in:** `report/checkpoint2.tex`, Sezione 5
**Script:** `experiment_v4.py`

**Idea chiave:** la rete riceve (x,y) = (Φ⁻¹(u), Φ⁻¹(v)) ∈ ℝ² anziché
(u,v) ∈ [0,1]². La copula diventa una funzione liscia nello spazio trasformato.

| Config | Transform | N | KL | err_max | L1_joint |
|---|---|---|---|---|---|
| ref_no_transform | No | 2k | 0.009 | 6.06 | 0.136 |
| transform_sp_2k | Sì | 2k | 0.007 | 6.25 | 0.128 |
| transform_sp_10k | Sì | 10k | **0.003** | 8.09 | 0.204 |
| transform_exp_10k | Sì | 10k | 0.019 | 243.6 | diverge |

**Risultato:** KL = 0.003, riduzione dell'88% rispetto alla naive (0.026).
La trasformazione è il singolo miglioramento più efficace. exp + trasformazione
diverge: la combinazione è catastrofica.

### A.7 — Ablation study: isolamento dei contributi ✅

**Documentato in:** `report/checkpoint2.tex`, Sezione 6
**Script:** `experiment_ablation.py`

Design fattoriale 2×2 su entrambe le baseline del checkpoint 1.
Unici fattori: N ∈ {2k, 10k} × transform ∈ {sì, no}. Tutto il resto identico al checkpoint 1.

| Config | KL | Δ vs baseline |
|---|---|---|
| naive | 0.024 | — |
| naive + Φ⁻¹ | **0.006** | -74% |
| naive + 10k | 0.026 | +8% |
| naive + Φ⁻¹ + 10k | **0.004** | -84% |
| best | 0.011 | — |
| best + Φ⁻¹ | 0.006 | -41% |
| best + 10k | 0.005 | -52% |
| best + Φ⁻¹ + 10k | **0.003** | -68% |

**Risultati chiave:**
- Φ⁻¹ è il fattore dominante: naive+Φ⁻¹ (KL=0.006, 4k params) batte best (KL=0.011, 34k params)
- Più dati senza trasformazione non aiutano il naive (λ=100 domina)
- I due fattori sono complementari
- Best complessivo: best+Φ⁻¹+10k, KL=0.003, L1_joint=0.115

---

## Fase B: Copule complesse (TODO)

- Clayton (dipendenza asimmetrica nella coda inferiore)
- Gumbel (dipendenza nella coda superiore)
- t-Student (code pesanti simmetriche)

## Fase C: Dataset reale (TODO)

- Applicazione a dati bivariati reali dove la vera copula è ignota
- Marginali stimate empiricamente (non più gaussiane note)

## Fase D: Confronto con baseline (TODO)

- KDE bivariata
- Gaussian Mixture Models
- Eventualmente copule parametriche classiche (MLE)
