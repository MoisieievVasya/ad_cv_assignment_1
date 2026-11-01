# %% [markdown]
# # Part 0 — EDA, metric, validation and input pipeline
#
# Hypotheses for this block (H0.1–H0.5) were committed in `HYPOTHESES.md` before this notebook was run.
# Everything heavy lives in `src/soilgs`; this notebook only calls it and shows the evidence.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from soilgs import paths
from soilgs.data import load_labels, load_photos, load_photo, crop_tray, resample
from soilgs.metric import (COLS, SIEVES, WEIGHTS, emd, emd_per_sample, emd_trapezoid_per_sample,
                           emd_per_interval, shift_curve, d_percentile, d_errors)
from soilgs.folds import make_folds, assign_sites
from soilgs.features import pixel_centroid
from soilgs.tiles import TARGET_PPM, load_cached, tile_grid
from soilgs.baselines import tile_feature_table, mean_curve_oof, ridge_tile_oof, ridge_tile_test
from soilgs.experiment import save_experiment, fold_scores, compute_log, mean_std
from soilgs.submission import write_submission

pd.set_option("display.width", 160)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
labels, photos = load_labels(), load_photos()
folds = pd.read_csv(paths.FOLDS_CSV)
Y = labels[COLS].to_numpy(float)
print(f"{labels.shape[0]} labelled samples, {len(photos.query('split==\"train\"'))} train photos, "
      f"{photos.query('split==\"test\"').sample_id.nunique()} test samples, {len(photos.query('split==\"test\"'))} test photos")

# %% [markdown]
# ## 1. EDA
# ### Samples, photos, devices, scale

# %%
dev = photos.groupby(["split", "camera"]).agg(photos=("photo_id", "size"), samples=("sample_id", "nunique"),
                                              native_ppm=("native_ppm", "first"), effective_ppm=("ppm", "mean"),
                                              long_side_px=("width", "max"))
dev["field_of_view_mm"] = (photos.groupby(["split", "camera"]).apply(lambda g: (g[["width", "height"]].max(axis=1) / g.ppm).mean())).round(0)
display(dev.round(2))
per_sample = photos.groupby(["split", "sample_id"]).agg(photos=("photo_id", "size"), devices=("camera", "nunique"))
display(per_sample.groupby("split").agg(["min", "median", "max"]))

# %% [markdown]
# **Takeaways.**
# * 24 labelled samples, not 127: photos of one sample are repeated views of the same soil.
# * The organisers downscaled the Motorola Edge and Samsung training photos, so they are **≈ 4.6 px/mm**,
#   while the native px/mm in `ppm_updated.csv` is 11.5–26.3. Test photos are native: **14–19.5 px/mm**.
#   Test images are therefore 3–4× sharper than anything the models are trained on. Everything below is
#   resampled to a common 4.5 px/mm (the finest scale every photo has).
# * Train and test use **disjoint devices** (Motorola/Samsung vs iPhone): device is confounded with split,
#   which is one more reason to make the input depend on millimetres, not pixels.

# %% [markdown]
# ### Curve shapes: fine vs coarse soils

# %%
lab = labels.set_index("sample_id")
lab["pct_lt_2mm"] = lab["2"]
lab["pct_lt_0.063mm"] = lab["0.063"]
lab["D50_mm"] = [d_percentile(c, 50) for c in Y]
lab["D90_mm"] = [d_percentile(c, 90) for c in Y]
lab["group"] = pd.cut(lab.pct_lt_2mm, [-1, 60, 90, 101], labels=["gravelly (<60% <2mm)", "mixed", "fine (>90% <2mm)"])

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
colors = {"gravelly (<60% <2mm)": "#b5651d", "mixed": "#7a7a7a", "fine (>90% <2mm)": "#2a6fbb"}
for sid, row in lab.iterrows():
    ax[0].plot(SIEVES, row[COLS].astype(float), color=colors[row.group], alpha=.7)
for g, c in colors.items():
    ax[0].plot([], [], color=c, label=f"{g} (n={(lab.group == g).sum()})")
ax[0].set(xscale="log", xlabel="sieve (mm)", ylabel="% passing", title="Training grain-size curves")
ax[0].legend(fontsize=8)
ax[1].bar(range(10), labels[COLS].std().to_numpy()[:-1], color="#555")
ax[1].set_xticks(range(10), [f"{a}–{b}" for a, b in zip(COLS[:-1], COLS[1:])], rotation=45, fontsize=8)
ax[1].set(title="Std of % passing at the left sieve of each interval", ylabel="std (points)")
plt.tight_layout(); plt.show()

s = np.linalg.svd(Y - Y.mean(0), compute_uv=False)
print("PCA variance explained by first 3 components:", (s**2 / (s**2).sum())[:3].round(3))
display(lab[["pct_lt_0.063mm", "pct_lt_2mm", "D50_mm", "D90_mm", "group"]].sort_values("D50_mm").round(2).T)

# %% [markdown]
# **Takeaways.** Most samples are fine (silty/sandy) soils; only a handful are gravelly. The curves are
# almost one-dimensional (one PC explains ~91 %), so the task is mostly "how coarse is it". Variance is
# concentrated at 0.02–2 mm, exactly the sizes a 4.6 px/mm photo resolves worst (one pixel ≈ 0.22 mm).

# %% [markdown]
# ## 2. Metric
# Competition metric: **EMD** = Σᵢ |F_true(dᵢ) − F_pred(dᵢ)| · (log₁₀dᵢ₊₁ − log₁₀dᵢ) over the 10 intervals
# (left-point rule). That is the area between the two cumulative curves on a log-diameter axis.
# Interval weights:

# %%
print(dict(zip([f"{a}-{b}" for a, b in zip(COLS[:-1], COLS[1:])], WEIGHTS.round(3))))
uniform = np.tile(np.cumsum([100 / 11] * 11), (len(Y), 1))
loo_mean = np.array([np.delete(Y, i, 0).mean(0) for i in range(len(Y))])
print(f"uniform baseline EMD = {emd(Y, uniform):.1f}   LOO mean-curve EMD = {emd(Y, loo_mean):.1f}  "
      "(public notebooks report 102 / 90 -> implementation matches)")
r = pd.Series(emd_per_sample(Y, loo_mean)).corr(pd.Series(emd_trapezoid_per_sample(Y, loo_mean)), method="spearman")
print(f"left-point vs trapezoid rule (one public notebook used the latter): mean {emd(Y, loo_mean):.1f} vs "
      f"{emd_trapezoid_per_sample(Y, loo_mean).mean():.1f}, Spearman over samples {r:.3f}")

# %% [markdown]
# ### H0.2 — Where does the error mass live?

# %%
contrib = pd.Series(emd_per_interval(Y, loo_mean).mean(0), index=[f"{a}–{b}" for a, b in zip(COLS[:-1], COLS[1:])])
share = contrib / contrib.sum()
display((share * 100).round(1).to_frame("% of mean-curve EMD").T)
print(f"0.02–2 mm intervals: {share.iloc[2:6].sum():.0%} of the EMD;  ≥ 20 mm intervals: {share.iloc[8:].sum():.1%}")

# %% [markdown]
# ### H0.1 — What does a scale error cost?
# A wrong px/mm by factor *k* makes every grain look *k* times bigger: F'(d) = F(d/k).

# %%
factors = [1.1, 1.25, 1.5, 2, 3, 4]
cost = pd.DataFrame({k: emd_per_sample(Y, np.array([shift_curve(y, k) for y in Y])) for k in factors}, index=labels.sample_id)
cost_down = emd_per_sample(Y, np.array([shift_curve(y, 0.5) for y in Y]))
display(cost.agg(["mean", "std", "min", "max"]).round(1))
print(f"theory 100·log10(k): {[round(100 * np.log10(k), 1) for k in factors]};  k = 0.5 (grains read 2x smaller): "
      f"{cost_down.mean():.1f} ± {cost_down.std():.1f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(factors, cost.mean(), "o-", label="mean over samples")
ax[0].plot(factors, 100 * np.log10(factors), "k--", label="100·log10(k)")
ax[0].axhline(emd(Y, loo_mean), color="r", ls=":", label="mean-curve baseline")
ax[0].set(xscale="log", xlabel="scale error factor k", ylabel="EMD", title="Cost of a scale error"); ax[0].legend()
ax[1].scatter(lab.loc[cost.index, "0.002"], cost[2], c=np.log10(lab.loc[cost.index, "D50_mm"]), cmap="viridis")
ax[1].set(xlabel="% passing 0.002 mm", ylabel="EMD of a 2× error", title="2× error cost vs share of clay")
plt.tight_layout(); plt.show()
print("corr(2x cost, % < 0.002 mm) =", round(np.corrcoef(lab.loc[cost.index, "0.002"], cost[2])[0, 1], 2),
      "; corr(2x cost, log D60/D10) =",
      round(np.corrcoef(np.log10([d_percentile(y, 60) / d_percentile(y, 10) for y in Y]), cost[2])[0, 1], 2))

# %% [markdown]
# ### Complementary and alternative metric
# * **Complementary: D10 / D50 / D60 error in log₁₀(mm).** Engineers use these numbers directly (e.g. the
#   uniformity coefficient Cu = D60/D10). An EMD of 30 can mean "the whole curve is one octave off" or
#   "the fines are 30 points wrong"; D-errors tell the two apart. Reported for every model alongside EMD.
# * **Alternative target metric: EMD on the *continuous* log-d axis with soil-class-weighted bands**, or
#   simply the mean absolute error of the four ISO 14688 fractions (clay < 0.002, silt 0.002–0.063,
#   sand 0.063–2, gravel > 2 mm). Justification: the left-point rule never looks at the 63–200 mm interval
#   (the error at 200 mm is always 0) and gives the clay share as much weight as a whole sand decade;
#   the fraction-MAE measures what a geotechnical classification actually uses and is insensitive to how
#   the sieves are spaced.

# %%
d_err = pd.DataFrame(d_errors(Y, loo_mean), columns=["D10", "D50", "D60"], index=labels.sample_id)
print("mean-curve baseline, |Δlog10 D| (1.0 = one decade off):"); display(d_err.mean().round(3).to_frame().T)

# %% [markdown]
# ## 3. Validation: site-grouped folds
# Photos of one sample never span train/validation; predictions are aggregated to one curve per sample
# before scoring. Folds also hold out whole **sites** (consecutive ids of one letter, e.g. H366–H374),
# because test samples come from sites absent from training. `data/folds.csv` is created once (seed 42,
# StratifiedGroupKFold stratified on D50 tercile) and used by **every** experiment.

# %%
display(folds.assign(D50=lambda d: d.sample_id.map(lab.D50_mm).round(3)).groupby("fold")
        .agg(samples=("sample_id", list), sites=("site", "nunique"), median_D50=("D50", "median")))

# %% [markdown]
# ### Baselines on the shared folds (logged to `experiments/`)
# Texture baseline: per-tile mm-indexed difference-of-Gaussian energies + Lab colour moments (23 features,
# `soilgs.features`), ridge (α = 10, fixed a priori, not tuned) on √mass targets, tile → photo → sample
# averaging.

# %%
with compute_log() as c:
    oof_mean = mean_curve_oof(labels, folds)
m = save_experiment("p0_mean_curve", {"model": "fold mean curve"}, oof_mean, labels, folds, c, part="0")
print(f"mean curve: CV {m['cv_mean']:.2f} ± {m['cv_std']:.2f}")

tiles = {t: tile_feature_table(photos, t) for t in (25, 50, 100)}
for t, df in tiles.items():
    print(f"{t:>3} mm tiles: {len(df)} tiles, {df.groupby('photo_id').size().median():.0f} per photo")

# %% [markdown]
# ### H0.4 — Tile size in millimetres

# %%
res = {}
for t in (25, 50, 100):
    with compute_log() as c:
        oof = ridge_tile_oof(tiles[t], labels, folds, alpha=10)
    cfg = {"model": "ridge", "alpha": 10, "target": "sqrt_mass", "features": "texture (23)", "tile_mm": t, "ppm": TARGET_PPM}
    m = save_experiment(f"p0_texture_ridge_{t}mm", cfg, oof, labels, folds, c, part="0")
    per_sample, _ = fold_scores(oof, labels, folds)
    res[t] = {"CV mean ± std (5 site folds)": f"{m['cv_mean']:.2f} ± {m['cv_std']:.2f}",
              "gravelly samples": per_sample[lab.loc[per_sample.index, "group"] == "gravelly (<60% <2mm)"].mean().round(2),
              "other samples": per_sample[lab.loc[per_sample.index, "group"] != "gravelly (<60% <2mm)"].mean().round(2)}
    if t == 50:
        oof_texture50 = oof
display(pd.DataFrame(res).T)

alpha_sweep = {a: emd(labels.set_index("sample_id").loc[o.index, COLS].to_numpy(float), o.to_numpy())
               for a in (1, 10, 100, 1000) for o in [ridge_tile_oof(tiles[50], labels, folds, alpha=a)]}
print("alpha sweep at 50 mm (reported, not used for selection):", {a: round(v, 2) for a, v in alpha_sweep.items()})

# %% [markdown]
# Tile size choice for the deep models: at 4.5 px/mm, **50 mm = 225 px**, the native input of ImageNet
# CNNs/ViTs at 224 px, so no resizing (which would change mm/px) is ever needed; 100 mm = 450 px is the
# "large context" alternative tested later as an input-resolution ablation. Training-set D90 is ≤ 20 mm for
# all but the coarsest samples, so a 50 mm tile holds several of the grains that carry the mass.

# %%
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, sid in zip(ax, ["F827", "H366", "H038"]):
    row = photos[(photos.sample_id == sid) & (photos.split == "train")].iloc[0]
    img = load_cached(row.photo_id)
    a.imshow(img)
    for y, x, t in tile_grid(*img.shape[:2], 50):
        a.add_patch(plt.Rectangle((x, y), t, t, fill=False, ec="yellow", lw=.6))
    a.set_title(f"{sid}: D50 = {lab.loc[sid, 'D50_mm']:.2g} mm, 50 mm grid", fontsize=9); a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### H0.3 — Site-grouped vs sample-grouped folds

# %%
rows = []
for group in ("site", "sample"):
    for seed in range(5):
        f = make_folds(labels, COLS, group=group, seed=seed)
        _, per_fold = fold_scores(ridge_tile_oof(tiles[50], labels, f, alpha=10), labels, f)
        rows.append({"grouping": group, "seed": seed, "cv_mean": per_fold.mean(), "fold_std": per_fold.std(ddof=0)})
grp = pd.DataFrame(rows)
display(grp.groupby("grouping")[["cv_mean", "fold_std"]].agg(["mean", "std"]).round(2))

# %% [markdown]
# ### H0.5 — The train/test resolution gap
# A pixel-indexed texture scale (DoG energy centroid in log₂ pixels, `features.pixel_centroid`) on each
# tray crop, at native resolution vs after resampling to 4.5 px/mm.

# %%
rows = []
for r in photos.itertuples():
    crop, _ = crop_tray(load_photo(r.path))
    rows.append({"split": r.split, "sample_id": r.sample_id, "native": pixel_centroid(crop),
                 "resampled": pixel_centroid(load_cached(r.photo_id))})
pc = pd.DataFrame(rows).groupby(["split", "sample_id"])[["native", "resampled"]].mean()
inside = {}
for col in ("native", "resampled"):
    lo, hi = pc.loc["train", col].min(), pc.loc["train", col].max()
    inside[col] = int(pc.loc["test", col].between(lo, hi).sum())
print("test samples inside the training range:", inside, "of", len(pc.loc["test"]))
fig, ax = plt.subplots(figsize=(7, 3))
for i, col in enumerate(("native", "resampled")):
    ax.scatter(pc.loc["train", col], np.full(24, i + .1), c="C0", label="train" if i == 0 else None)
    ax.scatter(pc.loc["test", col], np.full(len(pc.loc["test"]), i - .1), c="C3", label="test" if i == 0 else None)
ax.set_yticks([0, 1], ["native px", "4.5 px/mm"]); ax.set_xlabel("texture scale centroid (log2 px)"); ax.legend()
plt.tight_layout(); plt.show()
display(pc.loc["test"].round(2).T)

# %% [markdown]
# ## 4. First submission (texture baseline, 50 mm tiles)
# Fit on all 24 samples, predict test. Written to `submissions/`; it is submitted and its public/private LB
# recorded in the results table. The model was chosen by CV only.

# %%
test_curves = ridge_tile_test(tiles[50], labels, alpha=10)
sub = write_submission(test_curves, paths.ROOT / "submissions" / "p0_texture_ridge_50mm.csv")
display(sub.round(1))
