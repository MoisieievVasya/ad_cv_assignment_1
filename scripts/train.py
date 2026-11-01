"""Run one group-CV experiment from a YAML config.

    uv run python scripts/train.py configs/p1/resnet18_adamw.yaml
    uv run python scripts/train.py configs/p1/resnet18_adamw.yaml --set lr=1e-3 epochs=5 --name smoke --folds 0
    uv run python scripts/train.py configs/p1/convnext_t.yaml --test        # also fit on all data -> submission

A config file holds `name`, `part` and the training keys of `soilgs.train.DEFAULTS`; a file with an
`experiments:` list runs each entry (shared keys at top level).
"""
import argparse
from pathlib import Path

import yaml

from soilgs.train import run_cv


def parse_value(v):
    value = yaml.safe_load(v)
    if isinstance(value, str):  # YAML 1.1 reads "1e-5" (no dot) as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--name")
    ap.add_argument("--folds", help="comma-separated subset, e.g. 0,1 (smoke tests only)")
    ap.add_argument("--only", help="run only the experiment with this name from a multi-experiment file")
    ap.add_argument("--test", action="store_true", help="also train on all samples and write a submission")
    ap.add_argument("--save-checkpoints", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    shared = {k: v for k, v in spec.items() if k != "experiments"}
    runs = [{**shared, **e} for e in spec["experiments"]] if "experiments" in spec else [shared]
    overrides = dict(kv.split("=", 1) for kv in args.set)
    for run in runs:
        if args.only and run["name"] != args.only:
            continue
        cfg = {**run, **{k: parse_value(v) for k, v in overrides.items()}}
        name, part = args.name or cfg.pop("name"), str(cfg.pop("part", "?"))
        cfg.pop("name", None)
        folds = [int(f) for f in args.folds.split(",")] if args.folds else None
        run_cv(name, cfg, part, folds_to_run=folds, predict_test=args.test, save_checkpoints=args.save_checkpoints)


if __name__ == "__main__":
    main()
