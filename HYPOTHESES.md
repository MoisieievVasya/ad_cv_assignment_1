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
