"""Every experiment writes experiments/<name>/{config.yaml, oof.csv, metrics.json} so that the results
table, ensembling and the report are built from disk, never from notebook state."""
import json
import platform
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import yaml

from . import paths
from .metric import COLS, emd_per_sample


def device_info():
    try:
        import torch
        if torch.cuda.is_available():
            return {"device": torch.cuda.get_device_name(0)}
    except ImportError:
        pass
    return {"device": f"cpu ({platform.processor() or platform.machine()})"}


@contextmanager
def compute_log():
    """Measures wall time and peak GPU memory of the enclosed block into the yielded dict."""
    info = device_info()
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        cuda = False
    t0 = time.time()
    yield info
    info["hours"] = round((time.time() - t0) / 3600, 4)
    info["gpu_hours"] = info["hours"] if cuda else 0.0
    if cuda:
        info["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)


def fold_scores(oof, labels, folds):
    """Per-fold mean EMD of sample-level OOF curves (DataFrames indexed by sample_id)."""
    y = labels.set_index("sample_id").loc[oof.index, COLS].to_numpy(float)
    per_sample = pd.Series(emd_per_sample(y, oof[COLS].to_numpy(float)), index=oof.index)
    fold_of = folds.set_index("sample_id").fold
    return per_sample, per_sample.groupby(fold_of.loc[per_sample.index]).mean()


def save_experiment(name, config, oof, labels, folds, compute=None, extra=None, part=None):
    """Write config, OOF curves and metrics; return the metrics dict."""
    out = paths.EXPERIMENTS / name
    out.mkdir(parents=True, exist_ok=True)
    per_sample, per_fold = fold_scores(oof, labels, folds)
    metrics = {"name": name, "part": part, "cv_mean": round(float(per_fold.mean()), 3),
               "cv_std": round(float(per_fold.std(ddof=0)), 3),
               "oof_emd": round(float(per_sample.mean()), 3),
               "folds": [round(float(v), 3) for v in per_fold],
               "compute": compute or {}, **(extra or {})}
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    oof.assign(emd=per_sample).round(4).to_csv(out / "oof.csv")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def results_table():
    """All logged experiments as one DataFrame (the mandatory results table, LB columns filled by hand)."""
    rows = []
    for m in sorted(paths.EXPERIMENTS.glob("*/metrics.json")):
        d = json.loads(m.read_text())
        rows.append({k: d.get(k) for k in ("part", "name", "cv_mean", "cv_std", "oof_emd")}
                    | {"device": d.get("compute", {}).get("device"), "gpu_hours": d.get("compute", {}).get("gpu_hours"),
                       "peak_mem_gb": d.get("compute", {}).get("peak_mem_gb"),
                       "public_lb": d.get("public_lb"), "private_lb": d.get("private_lb")})
    return pd.DataFrame(rows)


def mean_std(values):
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}"
