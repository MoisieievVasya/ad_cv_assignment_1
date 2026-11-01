"""Write a valid submission from sample-level curves."""
import numpy as np
import pandas as pd

from . import paths
from .metric import COLS, project_valid


def write_submission(curves, out_path):
    """curves: DataFrame indexed by sample_id with COLS. Order and ids follow sample_submission.csv."""
    sub = pd.read_csv(paths.SAMPLE_SUBMISSION_CSV)
    sub["sample_id"] = sub.sample_id.astype(str)
    missing = set(sub.sample_id) - set(curves.index)
    if missing:
        raise ValueError(f"no prediction for {sorted(missing)}")
    values = project_valid(curves.loc[sub.sample_id, COLS].to_numpy(float))
    assert (np.diff(values, axis=1) >= -1e-9).all() and np.allclose(values[:, -1], 100)
    out = pd.DataFrame(values.round(4), columns=COLS)
    out.insert(0, "sample_id", sub.sample_id.values)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out
