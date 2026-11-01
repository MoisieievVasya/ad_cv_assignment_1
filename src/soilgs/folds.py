"""Group-aware folds shared by every experiment.

Grouping is by *site*, not only by sample: consecutive sample ids with the same letter (e.g. H366-H374)
come from one site and look alike, and the test samples come from sites absent from training.
Heuristic (same letter, numbers within 12 of the previous id) follows the public notebook
"soil-psd-physics-not-deep-learning".
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .metric import d_percentile

N_FOLDS = 5
FOLD_SEED = 42


def assign_sites(sample_ids, gap=12):
    ids = sorted(sample_ids, key=lambda s: (s[0], int(s[1:])))
    site, g, prev = {}, 0, None
    for s in ids:
        if prev is None or s[0] != prev[0] or int(s[1:]) - int(prev[1:]) > gap:
            g += 1
        site[s] = g
        prev = s
    return pd.Series(site, name="site")


def d50_stratum(labels, cols):
    """3 strata by D50 (fine / medium / coarse) so that every fold sees all soil types."""
    d50 = np.array([d_percentile(c, 50) for c in labels[cols].to_numpy(float)])
    return np.digitize(np.log10(d50), np.quantile(np.log10(d50), [1 / 3, 2 / 3]))


def make_folds(labels, cols, group="site", seed=FOLD_SEED, n_folds=N_FOLDS):
    """DataFrame sample_id, site, stratum, fold. `group` is 'site' or 'sample'."""
    df = labels[["sample_id"]].copy()
    df["site"] = df.sample_id.map(assign_sites(df.sample_id))
    df["stratum"] = d50_stratum(labels, cols)
    groups = df.site if group == "site" else df.sample_id
    df["fold"] = -1
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(df, df.stratum, groups)):
        df.loc[df.index[va], "fold"] = k
    return df
