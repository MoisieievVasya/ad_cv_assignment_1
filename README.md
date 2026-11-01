# Soil grain-size distribution from photos — AI401 Assignment #1

Kaggle competition: *Predicting Soil Grain Size Distributions from Images*
(`soil-grain-size-from-photos`). Work in progress; the report is filled in as blocks finish.

- `HYPOTHESES.md` — hypotheses log (committed before each block of experiments).
- `src/soilgs/` — library code (data, metric, folds, tiling, models, training).
- `scripts/` — command-line entry points.
- `notebooks/` — analysis notebooks, one per part.
- `configs/` — experiment configs; `experiments/<name>/` — OOF, metrics and compute log per run.

## Setup

```bash
uv sync
export KAGGLE_API_TOKEN=...   # your own token
uv run kaggle competitions download soil-grain-size-from-photos -p data/raw && (cd data/raw && unzip -q *.zip && rm *.zip)
```
