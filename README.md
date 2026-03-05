# Copula ANN Experiments (Checkpoint 1-2)

Questo repository contiene gli esperimenti sulla stima della densità di copula gaussiana con MLP.

## Obiettivo di riproducibilità

Ogni script ora salva gli artefatti in una directory dedicata:

- `outputs/<timestamp>_<script>/metadata.json`
- `outputs/<timestamp>_<script>/results.json`
- `outputs/<timestamp>_<script>/plots/*.png`
- `outputs/<timestamp>_<script>/best_model.pt` (dove previsto)

Questo evita sovrascritture tra run diverse.

## Compatibilità con artefatti legacy

Per continuare a scrivere anche nei path storici (`plots/` e `best_model.pt` in root), usare:

```bash
WRITE_LEGACY_ARTIFACTS=1 python3 experiment_v4.py
```

Default: `WRITE_LEGACY_ARTIFACTS=0` (solo `outputs/`).

## Dipendenze Python

Minimo richiesto:

- `python>=3.10`
- `torch`
- `numpy`
- `scipy`
- `matplotlib`

## Script e scopo

- `run.py`: demo baseline del checkpoint 1 (`softplus`, no transform)
- `grid_search.py`: grid search checkpoint 1
- `gen_all_plots.py`: plot checkpoint 1 (naive + best)
- `experiment_v2.py`: confronto attivazione/dati/scheduler
- `experiment_v3.py`: exp + clipping + dataset grandi
- `experiment_v4.py`: trasformazione input `Phi^-1`
- `experiment_ablation.py`: ablation 2x2 su naive e best

## Comandi tipici

```bash
python3 run.py
python3 grid_search.py
python3 gen_all_plots.py
python3 experiment_v2.py
python3 experiment_v3.py
python3 experiment_v4.py
python3 experiment_ablation.py
```

## Note metodologiche

Le integrazioni numeriche (penalty e valutazione KL) sono calcolate sul dominio
`[0.01, 0.99]^2` per stabilità numerica, non su `[0,1]^2` con estremi inclusi.

## LaTeX report

I report sono in `report/`:

- `checkpoint1.tex`
- `checkpoint2.tex`

Compilazione:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error checkpoint1.tex
pdflatex -interaction=nonstopmode -halt-on-error checkpoint2.tex
```

I sorgenti usano fallback lingua (italiano se disponibile, altrimenti inglese).
Se compare un errore babel su `language nil`, azzerare i file ausiliari del report (`checkpoint*.aux`, `checkpoint*.out`) e ricompilare.
