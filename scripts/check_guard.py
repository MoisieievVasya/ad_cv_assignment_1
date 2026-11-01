"""H6c: leave-coarsest-out check of the extrapolation guard (decision rule fixed in HYPOTHESES.md).

    uv run python scripts/check_guard.py
"""
import numpy as np
import pandas as pd

from soilgs import paths
from soilgs.baselines import ridge_tile_test, tile_feature_table
from soilgs.data import load_labels, load_photos
from soilgs.guard import Guard
from soilgs.metric import COLS, emd_per_sample

labels, photos = load_labels(), load_photos()
lab = labels.set_index("sample_id")
tiles = tile_feature_table(photos, 50)
cent = tiles.groupby(["split", "sample_id"]).dog_centroid.mean()
tr_c = cent.loc["train"]
rows = []
for s in tr_c.sort_values().index[-5:]:
    keep = tr_c.index[tr_c <= tr_c[s] - 0.15]
    sub_tiles = tiles[(tiles.split == "train") & tiles.sample_id.isin(list(keep) + [s])].copy()
    sub_tiles.loc[sub_tiles.sample_id == s, "split"] = "test"          # held out, "beyond range"
    main = ridge_tile_test(sub_tiles, labels[labels.sample_id.isin(keep)], alpha=10).loc[[s], COLS].to_numpy()
    g = Guard().fit(tr_c[keep].to_numpy(), lab.loc[keep, COLS].to_numpy(float))
    guarded, w = g.apply(tr_c[s], main)
    y = lab.loc[[s], COLS].to_numpy(float)
    rows.append({"sample": s, "centroid": round(tr_c[s], 2), "train max": round(g.c_max, 2), "n train": len(keep),
                 "w": round(float(w[0]), 2), "% <2mm": lab.loc[s, "2"], "EMD main": emd_per_sample(y, main)[0],
                 "EMD main+guard": emd_per_sample(y, guarded)[0]})
df = pd.DataFrame(rows)
df["gain"] = df["EMD main"] - df["EMD main+guard"]
print(df.round(2).to_string(index=False))
print(f"mean gain {df.gain.mean():.2f}, improved on {(df.gain > 0).sum()}/{len(df)} -> "
      f"{'USE' if df.gain.mean() >= 3 and (df.gain > 0).sum() >= 3 else 'DO NOT USE'} the guard (rule from H6c)")
(paths.ROOT / "reports").mkdir(exist_ok=True)
df.to_csv(paths.ROOT / "reports" / "guard_check.csv", index=False)
full = Guard().fit(tr_c.to_numpy(), lab.loc[tr_c.index, COLS].to_numpy(float))
print("guard weight on test samples:", dict(zip(cent.loc["test"].index, full.weight(cent.loc["test"].to_numpy()).round(2))))
