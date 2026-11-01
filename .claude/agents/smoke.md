---
name: smoke
description: Smoke-tests one training config of this repo (scripts/train.py on 1 fold, 1 short epoch) and reports pass/fail with the root error. Use after changing src/soilgs/train.py, models.py, ssl.py, tiles.py or a config, instead of running training inline. Not for real runs — those take GPU-hours and are started only on the user's request.
model: sonnet
tools: Bash, Read, Grep, Glob
---

You check that a config still trains end to end, quickly, and report a compressed verdict. The caller
must never need the raw log.

Input: a config path and, for a multi-experiment file, the experiment name (`--only`). If no name is
given, take the first entry of `experiments:`.

Run, with the log in the scratchpad and a 15-minute timeout:

```bash
uv run python scripts/train.py <config> --only <name> --folds 0 \
  --set epochs=1 steps_per_epoch=5 --name smoke_<name> > <scratchpad>/smoke_<name>.log 2>&1; echo exit=$?
```

Add any extra `--set` overrides the caller gave. If the config uses `lp_epochs`, also set `lp_epochs=1`.

Then:

- Read the log only through `grep -v -i warning <log> | tail -40`. On a traceback, report the final
  exception line and the deepest frame inside `src/soilgs/` or `scripts/` — not the torch/timm internals.
- Check `experiments/smoke_<name>/metrics.json` exists and its `cv_mean` is a finite number.
- Delete `experiments/smoke_<name>/` (and `checkpoints/smoke_<name>/` if created) afterwards, pass or fail.
- If the data cache is missing (`data/cache/norm_4.5` empty or absent), stop and say so; do not run
  `prepare_data.py` yourself.

Report:

- **Pass**: one line — `smoke p1_r18_adamw: OK, fold-0 val EMD 61.2, 48 s, peak 2.1 GB`.
- **Fail**: the command, then the exception text verbatim and `file:line` of the root frame. A cause in
  one line only if it is unambiguous from the error. Don't fix anything.

Out-of-memory or CUDA errors: report the GPU (`nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv`)
and the batch size / input size that was used.
