# Hypotheses log

Rules for this file:
- Each block's hypotheses and predicted outcomes are committed **before** the experiments that
  test them (check `git log -p HYPOTHESES.md`).
- After the block, the *Result* and *Why* parts are added in a separate commit. Predictions are
  never edited after the fact; wrong ones stay and get explained.
- Scores are the competition EMD (lower is better) on the shared site-grouped folds unless stated.

Prior knowledge used when writing these (public Kaggle notebooks, cited in the README): uniform
baseline ≈ 102, mean-curve baseline ≈ 90, hand-crafted mm-indexed texture + ridge ≈ 38–41,
frozen DINOv2-S + ridge on √mass targets ≈ 35. Training photos were downscaled by the organisers to
≈ 4.6 px/mm; test photos are native (14–19.5 px/mm).

---

## Block 0 — EDA, metric, validation, input pipeline

### H0.1 — What a 2× scale error costs
**Hypothesis.** A scale error by factor *k* shifts the cumulative curve by log10(k) along the
log-diameter axis. The metric is (up to discretisation) the area between the curves in log10(mm)
× percent, so for a curve that rises from 0 to 100 % inside the sieve range the cost of a pure
shift is ≈ 100 · log10(k), independent of the curve shape.
**Prediction.** A 2× error costs ≈ 30 EMD (100 · 0.301) on average, i.e. one third of the
mean-curve baseline. It is *cheaper* for fine soils, whose curve starts well above 0 % at 0.002 mm
(part of the mass lies below the support and does not move), and roughly shape-independent
otherwise. Competing intuition (from the first discussion): steep, uniformly graded curves cost
more. I expect the discrete left-point sum to make the cost vary by only ±10 EMD between samples,
so that intuition should *not* hold on average.
**Experiment.** Shift every training curve by ±log10(2) (interpolation in log-d), score against
the original, and plot cost vs. % < 0.002 mm and vs. curve steepness (D60/D10).

### H0.2 — Where the error mass lives
**Hypothesis.** Most target variance sits at 0.063–2 mm (std 28–38 points there vs < 8 at
≥ 20 mm), and the per-interval weights are almost equal (≈ 0.5 each), so the 0.02–2 mm intervals
dominate any EMD.
**Prediction.** For the mean-curve baseline, the four intervals between 0.02 and 2 mm carry
≥ 60 % of the EMD; the two intervals above 20 mm carry < 5 %. This also means the
silt/fine-sand split, the part photos resolve worst, decides the score.
**Experiment.** Per-interval decomposition of the leave-one-out mean-curve EMD.

### H0.3 — Site-grouped vs sample-grouped validation
**Hypothesis.** Several training samples come from the same site (e.g. H366–H374) and look alike.
Holding out a sample while its siblings stay in training leaks site information.
**Prediction.** With the same simple model (ridge on mm-indexed texture-band energies), CV with
folds grouped by *site* is worse than CV grouped only by *sample* by 1–4 EMD, and the std across
the 5 site-grouped folds is large (≥ 5 EMD), comparable to the gaps between the models we will
compare later.
**Experiment.** Same model and features; StratifiedGroupKFold(5) with groups = site vs groups =
sample, 5 repetitions with different seeds each.

### H0.4 — Tile size in millimetres
**Hypothesis.** A tile must contain several of the largest grains that carry mass. The D90 of
gravelly training samples is 10–40 mm, so a 25 mm tile can't see the gravel fraction, while
larger tiles average over more grains.
**Prediction.** At a fixed 4.5 px/mm, 50 mm and 100 mm tiles are within 1 std of each other;
25 mm tiles are worse by ≥ 3 EMD, and the loss comes mostly from coarse samples (% < 2 mm < 60).
**Experiment.** The same texture-ridge baseline computed per tile and averaged per sample, for
tile sizes 25 / 50 / 100 mm.

### H0.5 — The train/test resolution gap
**Hypothesis.** Because training photos are ≈ 4.6 px/mm and test photos are 14–19.5 px/mm,
any pixel-indexed statistic is out of distribution on test unless test is resampled to the
training px/mm first.
**Prediction.** For a pixel-indexed texture scale statistic (energy centroid of a
difference-of-Gaussians stack in *pixels*), ≤ 3 of 10 test samples fall inside the training range
when test photos are used at native resolution, and ≥ 8 of 10 when they are resampled to
4.5 px/mm. (The Münster cobble sample may stay outside: it is coarser than anything in training.)
**Experiment.** Compute the statistic on all photos both ways and count test samples inside
[min, max] of the training samples.

### Block 0 — results (notebooks/00_eda_metric_validation.ipynb)

| | Prediction | Result | Verdict |
|---|---|---|---|
| H0.1 | 2× scale error ≈ 30 EMD, ±10 between samples, cheaper for clay-rich soils; steep curves *not* costlier | 29.8 ± 0.5 (27.8–30.1); k = 0.5 → 29.2; 100·log10(k) matches for every k; corr with % < 0.002 mm −0.28, with log(D60/D10) −0.39 | **Confirmed**, and tighter than predicted |
| H0.2 | 0.02–2 mm intervals ≥ 60 % of mean-curve EMD, ≥ 20 mm < 5 % | 62 % and 4.4 %; 2–6.3 mm alone is another 17 % | **Confirmed** |
| H0.3 | site-grouped CV worse than sample-grouped by 1–4 EMD, fold std ≥ 5 | 40.97 ± 1.55 vs 36.47 ± 1.06 over 5 seeds (gap 4.5); fold std 13.1 vs 9.3 | **Confirmed** (gap slightly above the range) |
| H0.4 | 25 mm tiles ≥ 3 EMD worse than 50/100 mm, loss on gravelly samples | 38.55 / 38.17 / 38.15; gravelly samples 33.7 / 31.4 / 30.7 | **Wrong** |
| H0.5 | ≤ 3/10 test samples in train range at native px, ≥ 8/10 after resampling | 8/10 native, 7/10 resampled | **Wrong as measured**, see below |

**Why H0.4 was wrong.** The texture baseline averages per-tile statistics over the whole photo
(88 tiles of 25 mm vs 8 of 100 mm), so the photo-level feature is almost the same whatever the tile
size: a 30 mm stone that does not fit into one 25 mm tile still shows up as strong 8–16 mm DoG
energy in the tiles that cut through it. Tile size matters for *what one tile can see*, which
matters for a model that predicts from one tile at a time (the CNN/ViT), not for averaged
hand-crafted statistics. The question is re-tested for the deep models (input context ablation in
Part 1). Decision: 50 mm tiles (= 225 px at 4.5 px/mm, the native input size of ImageNet models,
no resizing).

**Why H0.5 was wrong — and what it showed instead.** "Inside the training range" was the wrong
test. At native resolution the test sands have a pixel texture scale of 2.69–2.89 log2 px, the
same as the training *gravels* H181, H183, H371, H615 (only 23–37 % < 2 mm; across training
samples the pixel centroid correlates −0.88 with % < 2 mm). So a pixel-indexed model would see
the test sands as in-distribution and call them gravel. After resampling to 4.5 px/mm the sands
move to 1.75–2.32, next to the fine training soils. Also, the Münster cobbles (3.55) and Testfeld
(2.94) stay at the coarse end in both versions. This is the strongest argument for the mm-calibrated pipeline:
the risk is not out-of-range inputs but in-range inputs that mean something else.

**Other Block 0 facts used later.** Train and test devices are disjoint (Motorola/Samsung vs
iPhone), test photos are 3–4× sharper than training photos. The mean-curve baseline on the shared
folds is 93.4 ± 9.6. The texture ridge reaches 38.2 ± 14.6. The left-point and trapezoid metric
variants rank samples identically (Spearman 0.999).
