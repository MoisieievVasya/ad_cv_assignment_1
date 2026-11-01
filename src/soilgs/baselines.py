"""Non-deep baselines: mean curve and ridge on per-tile texture features.

Tile-level rows inherit their sample's curve; predictions are made per tile in the encoded target
space, averaged tile -> photo -> sample, then decoded (see `tiles.aggregate`).
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import paths
from .features import FEATURE_NAMES, texture_features
from .metric import COLS
from .targets import decode, encode
from .tiles import TARGET_PPM, iter_tiles, load_cached


def _photo_tiles(args):
    photo_id, sample_id, split, tile_mm = args
    img = load_cached(photo_id)
    return [dict(photo_id=photo_id, sample_id=sample_id, split=split, tile=i,
                 **dict(zip(FEATURE_NAMES, texture_features(t, TARGET_PPM))))
            for i, t in enumerate(iter_tiles(img, tile_mm))]


def tile_feature_table(photos, tile_mm, workers=8):
    """Texture features for every tile of every cached photo (cached as pickle per tile size)."""
    path = paths.CACHE / f"texture_tiles_{tile_mm:g}mm.pkl"
    if path.exists():
        return pd.read_pickle(path)
    jobs = [(r.photo_id, r.sample_id, r.split, tile_mm) for r in photos.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        df = pd.DataFrame([row for rows in ex.map(_photo_tiles, jobs) for row in rows])
    df.to_pickle(path)
    return df


def mean_curve_oof(labels, folds):
    """Fold-wise mean training curve (the trivial baseline)."""
    lab = labels.set_index("sample_id")
    fold_of = folds.set_index("sample_id").fold
    oof = {}
    for k in sorted(fold_of.unique()):
        tr = fold_of.index[fold_of != k]
        mean = lab.loc[tr, COLS].to_numpy(float).mean(0)
        for s in fold_of.index[fold_of == k]:
            oof[s] = mean
    return pd.DataFrame(oof, index=COLS).T.loc[lab.index]


def ridge_tile_oof(tiles, labels, folds, alpha=10.0, mode="sqrt_mass", feature_cols=FEATURE_NAMES,
                   fold_col="fold"):
    """OOF sample curves from a tile-level ridge; folds is sample_id -> `fold_col`."""
    lab = labels.set_index("sample_id")
    df = tiles[tiles.split == "train"].merge(folds[["sample_id", fold_col]], on="sample_id")
    Z = encode(lab.loc[df.sample_id, COLS].to_numpy(float), mode)
    X = df[list(feature_cols)].to_numpy(float)
    preds = np.zeros_like(Z)
    for k in sorted(df[fold_col].unique()):
        tr, va = (df[fold_col] != k).to_numpy(), (df[fold_col] == k).to_numpy()
        sc = StandardScaler().fit(X[tr])
        preds[va] = Ridge(alpha=alpha).fit(sc.transform(X[tr]), Z[tr]).predict(sc.transform(X[va]))
    return _to_sample_curves(df, preds, mode)


def ridge_tile_test(tiles, labels, alpha=10.0, mode="sqrt_mass", feature_cols=FEATURE_NAMES):
    """Fit on all training tiles, predict test samples."""
    lab = labels.set_index("sample_id")
    tr, te = tiles[tiles.split == "train"], tiles[tiles.split == "test"]
    sc = StandardScaler().fit(tr[list(feature_cols)].to_numpy(float))
    model = Ridge(alpha=alpha).fit(sc.transform(tr[list(feature_cols)].to_numpy(float)),
                                   encode(lab.loc[tr.sample_id, COLS].to_numpy(float), mode))
    return _to_sample_curves(te, model.predict(sc.transform(te[list(feature_cols)].to_numpy(float))), mode)


def _to_sample_curves(df, preds, mode):
    zcols = [f"z{i}" for i in range(preds.shape[1])]
    agg = pd.concat([df[["photo_id", "sample_id"]].reset_index(drop=True), pd.DataFrame(preds, columns=zcols)], axis=1)
    sample_z = agg.groupby(["photo_id", "sample_id"])[zcols].mean().groupby("sample_id").mean()
    return pd.DataFrame(decode(sample_z.to_numpy(), mode), index=sample_z.index, columns=COLS)
