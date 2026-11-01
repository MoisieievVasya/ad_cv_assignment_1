# soilgs — working notes for Claude

Kaggle "soil grain-size from photos" (AI401 Assignment #1). Library in `src/soilgs/`, entry points in
`scripts/`, experiment grids in `configs/`, results on disk in `experiments/<name>/`. All Python runs via
`uv run`.

## Where the bytes are — and what never goes into context

The code is ~2k lines; everything heavy sits next to it. Query these, never `Read` them:

| Path | What | How to look instead |
|---|---|---|
| `data/raw/` (378 MB) | competition photos + label CSVs | `head -3`, `wc -l`, pandas one-liner |
| `data/cache/` (400 MB) | 4.5 px/mm `.png` photos + texture `.pkl` | never; it is rebuilt by `scripts/prepare_data.py` |
| `notebooks/*.ipynb` | executed notebooks, base64 figures (~1 MB) | edit `notebooks/src/*.py`; read outputs with `jq` (below) or the `notebook` agent |
| `.venv/`, `uv.lock` | env / lockfile | `uv tree`, `rg -c <pkg> uv.lock` |
| `checkpoints/`, `*.pt` | weights | `ls -la` only |
| `experiments/*/oof.csv`, `history.csv` | per-sample curves | `jq`/pandas for aggregates; `metrics.json` holds the scores |

Photos are images: one accidental `Read` of a photo costs more than reading the whole library.

Small files worth reading whole when relevant: `src/soilgs/*.py` (each < 330 lines), `configs/**/*.yaml`,
`scripts/*.py`. `HYPOTHESES.md` (~14 KB) — read by section: `grep -n '^##' HYPOTHESES.md`, then
`Read` with `offset`/`limit`.

## Commands

```bash
uv run python scripts/prepare_data.py                             # photo index + folds + cache (once)
uv run python scripts/train.py configs/p1/cnn.yaml --only p1_r18_adamw         # one run of a grid
uv run python scripts/train.py <cfg> --only <name> --folds 0 \
    --set epochs=1 steps_per_epoch=5 --name smoke_<name>          # smoke test, ~1 min
uv run python scripts/build_notebook.py notebooks/src/00_eda_metric_validation.py   # executes + writes .ipynb
```

No test suite yet (`pyproject` points pytest at a missing `tests/`).

## Long-running work

- A full grid or even one 5-fold run is GPU-hours. Never run it in the foreground and never start one
  without the user asking — offer a smoke test instead. Run real training with `run_in_background`,
  stdout to a log file, and read only `tail -20` / `grep '\] \(fold\|CV\)'` of it when notified.
- Training prints one line per fold and one CV line; timm/torch warnings are the noise. Filter with
  `grep -v Warning` before showing anything.
- Smoke runs write `experiments/smoke_*/`; delete that directory afterwards so it doesn't enter
  `results_table()`.
- Cloud runs (Colab/Kaggle via `notebooks/src/colab_runner.py`) push to `results/<grid>` branches;
  inspect with `git show --stat origin/results/...`, not by checking them out.

## Reading results cheaply

```bash
jq -c '{name,cv_mean,cv_std,oof_emd}' experiments/*/metrics.json                 # all runs, one line each
uv run python -c "from soilgs.experiment import results_table as r; print(r().to_string())"
```

EMD: lower is better. Compare runs by `cv_mean` with `cv_std` in mind — fold std is ~15 EMD, so
differences under a few points are noise.

Notebook outputs without the images:

```bash
NB=notebooks/00_eda_metric_validation.ipynb
jq -r '.cells | to_entries[] | select(.value.cell_type=="code") | "## cell \(.key)", (.value.outputs[]? | select(.output_type=="stream") | .text | if type=="array" then join("") else . end)' $NB | head -100
jq -r '.cells[].outputs[]? | select(.output_type=="error") | "\(.ename): \(.evalue)"' $NB
```

## Delegate instead of reading

| Question | Agent |
|---|---|
| "which runs beat X", "results table for block N", "per-fold scores of …" | `results` (haiku) |
| "does this config / code change still train" | `smoke` (sonnet) |
| "rebuild notebook N", "what did cell K print", "did the notebook break" | `notebook` (sonnet) |
| "where is X defined / used" | `locate` (haiku, user-level) |
| "when / why did X change", `HYPOTHESES.md` history | `history` (haiku, user-level) |

Do it inline when it is one `jq` / `rg` / short `Read` — an agent round trip costs more than that.

## Project rules

- `HYPOTHESES.md`: predictions for a block are committed before its experiments; *Result* / *Why* are
  added later in a separate commit. Never edit a prediction after the fact.
- `data/folds.csv` and `data/photos.csv` are frozen in git; `git status data/` must stay clean after
  `prepare_data.py`.
- Every experiment writes `experiments/<name>/{config.yaml, oof.csv, metrics.json, history.csv}`; tables
  and the report are built from those files, never from notebook state.
- `notebooks/src/*.py` (percent format) is the reviewed source; the `.ipynb` is a build output.
