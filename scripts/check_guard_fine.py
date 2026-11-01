"""H6d: leave-finest-out check of the fine-side extrapolation guard (decision rule fixed in HYPOTHESES.md).

    uv run python scripts/check_guard_fine.py
"""
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
for s in tr_c.sort_values().index[:5]:
    keep = tr_c.index[tr_c >= tr_c[s] + 0.15]
    sub = tiles[(tiles.split == "train") & tiles.sample_id.isin(list(keep) + [s])].copy()
    sub.loc[sub.sample_id == s, "split"] = "test"
    main = ridge_tile_test(sub, labels[labels.sample_id.isin(keep)], alpha=10).loc[[s], COLS].to_numpy()
    g = Guard(two_sided=True).fit(tr_c[keep].to_numpy(), lab.loc[keep, COLS].to_numpy(float))
    guarded, w = g.apply(tr_c[s], main)
    y = lab.loc[[s], COLS].to_numpy(float)
    rows.append({"sample": s, "centroid": round(tr_c[s], 2), "train min": round(g.c_min, 2), "n train": len(keep),
                 "w": round(float(w[0]), 2), "% <0.063mm": lab.loc[s, "0.063"], "EMD main": emd_per_sample(y, main)[0],
                 "EMD main+guard": emd_per_sample(y, guarded)[0]})
df = pd.DataFrame(rows)
df["gain"] = df["EMD main"] - df["EMD main+guard"]
print(df.round(2).to_string(index=False))
use = df.gain.mean() >= 3 and (df.gain > 0).sum() >= 3
print(f"mean gain {df.gain.mean():.2f}, improved on {(df.gain > 0).sum()}/{len(df)} -> {'USE' if use else 'DO NOT USE'} (rule from H6d)")
df.to_csv(paths.ROOT / "reports" / "guard_fine_check.csv", index=False)
full = Guard(two_sided=True).fit(tr_c.to_numpy(), lab.loc[tr_c.index, COLS].to_numpy(float))
print("two-sided guard weight on test samples:", {k: round(float(v), 2) for k, v in zip(cent.loc["test"].index, full.weight(cent.loc["test"].to_numpy())) if v > 0})
