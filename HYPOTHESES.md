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

---

## Block 1 — CNNs (configs/p1/cnn.yaml)

Common setup: tiles of 50 mm at 4.5 px/mm (225 px), random tile positions, softmax over 11 bins →
cumulative curve, EMD loss, 15 × 100 steps, batch 32, grid-tile + flip TTA at validation,
tile → photo → sample averaging. Reference points: texture ridge 38.2 ± 14.6, mean curve 93.4.

### H1.1 — Pretrained CNN vs hand-crafted texture, and architecture
**Hypothesis.** With 19 training samples per fold, a fine-tuned CNN learns roughly the same
"dominant texture scale" signal as the DoG features; it cannot learn much more.
**Prediction.** ResNet-18 (AdamW baseline) lands at 34–42, within one fold-std of the texture
ridge. ResNet-50 ≈ ResNet-18 (± 2); ConvNeXt-T (IN-22k) is the best CNN but by less than 4 EMD;
EfficientNet-B0 ≈ ResNet-18.

### H1.2 — Optimiser and schedule
**Hypothesis.** For a pretrained, BatchNorm-equipped CNN the loss landscape is benign: SGD and AdamW
reach similar solutions and warm-up is unnecessary.
**Prediction.** SGD (lr 0.01) vs AdamW (3e-4): |Δ| < 2 EMD. Removing warm-up: |Δ| < 1. AdamW at
lr 1e-3 is worse by ≥ 2 (the pretrained filters get overwritten on 19 samples).

### H1.3 — BatchNorm when a batch holds few physical samples
**Hypothesis.** The label is a property of the whole sample, and much of it is in the *absolute*
level of texture statistics (how much high-frequency energy, how bright). Train-mode BN normalises
each channel by the batch statistics; if a batch contains tiles of only 2 samples, it removes exactly
that between-sample level difference during training, and the running statistics used at test
time describe a different distribution.
**Prediction.** Batches drawn from 2 samples (`k2`) with train-mode BN are worse than random batches
by ≥ 3 EMD. Frozen BN (ImageNet statistics) closes the gap: frozen-BN-k2 ≈ frozen-BN ≈ random-BN
(± 1.5). GroupNorm, swapped into a BN-pretrained network, is worse than frozen BN by 1–3
(the pretrained weights were calibrated to BN statistics).

### H1.4 — Augmentations
**Hypothesis.** Flips and 90° rotations do not change grain size (texture is isotropic); random tile
positions already augment a lot. Scale augmentation changes mm/px and therefore the correct answer;
blur erases the finest grains.
**Prediction.** No flips/rotations: worse by ≥ 1. Colour jitter (± 20 %): neutral (|Δ| < 1.5).
Random rescale ×[0.5, 2]: worse by ≥ 5 (Block 0: one octave of scale noise costs ~30 EMD per
sample, and the model can only learn the scale-averaged answer). Blur (σ up to 2 px ≈ 0.45 mm):
worse by ≥ 2, mostly on fine soils.

### H1.5 — Input resolution and context
**Prediction.** Halving resolution (50 mm tile at 112 px = 2.25 px/mm) is worse by ≥ 3, mostly on
fine soils. Doubling context (100 mm tile, 450 px) is within ± 2 overall but better on gravelly
samples, the opposite of what H0.4 showed for averaged hand-crafted statistics, because a single
CNN tile has to see whole stones.

### H1.6 — Loss
**Prediction.** Training with the EMD itself beats sqrt-mass MSE by ≥ 2 EMD.

### H1.7 — Pretraining for CNNs
**Hypothesis.** The CNN inductive biases (locality, weight sharing, translation equivariance,
pooling) suit stationary texture, so even from scratch a CNN learns something; but 19 samples
are far too few to learn good filters.
**Prediction.** ResNet-18 from scratch: 50–75 EMD, i.e. clearly better than the mean curve (93) but
≥ 15 worse than pretrained. Later contrast: a from-scratch ViT will be much closer to the mean curve
(Block 2).

---

## Block 2 — Vision Transformers (configs/p2/vit.yaml)

Written before any Block 1 result was known (the CNN grid was still running), so these
predictions do not condition on the CNN numbers.

### H2.1 — From scratch vs pretrained ViT vs pretrained CNN
**Hypothesis.** A ViT has almost no image prior: it must learn locality and translation equivariance
from data. 19 samples (≈ 100 photos) cannot teach that, so a from-scratch ViT learns at most the
global colour/brightness of a tile. Pretraining supplies the prior that a CNN gets from its
architecture.
**Prediction.** Order: pretrained CNN (ResNet-18) ≈ pretrained ViT-S (within ± 4) ≫ own ViT from
scratch. The scratch ViT lands at 60–85, close to colour-only predictors and far from the
pretrained models; patch 8 does not save it (± 5 vs patch 16). The pretrained-vs-scratch gap is
larger for the ViT than for ResNet-18 (H1.7).

### H2.2 — Warm-up and learning rate
**Hypothesis.** Adam on a ViT has large, poorly-estimated second moments in the first steps, and
attention logits can blow up; LayerNorm-only nets lack BN's implicit scale control.
**Prediction.** Removing warm-up hurts the ViT (≥ 2 EMD) more than it hurt ResNet-18 (H1.2 predicted
< 1). lr 1e-3 is clearly harmful (≥ 5 worse). SGD is worse than AdamW by ≥ 5 (vs < 2 for the CNN).

### H2.3 — Layer-wise LR decay and weight decay placement
**Prediction.** LLRD 0.75 beats no LLRD by ≥ 1.5 (protects the generic early blocks on a tiny
dataset). Applying weight decay to LayerNorm/bias/position embeddings is within ± 1 (wd 0.05 over
1500 steps is too weak to matter much).

### H2.4 — Stochastic depth
**Prediction.** 0.1 is best; 0 vs 0.1 within ± 1.5; 0.3 is worse by ≥ 1.5 (too much noise for a
6-block-equivalent-depth adaptation with 1500 steps).

### H2.5 — Patch size in millimetres, architecture, pretraining data
**Prediction.** 384 px input (2.1 mm patches) beats 224 px (3.6 mm patches) by ≥ 2, mostly on
fine soils. DeiT-S (IN-1k) is worse than ViT-S IN-21k by ≥ 2. Swin-T is the best transformer
(local windows ≈ CNN prior) and lands within ± 2 of the best CNN; MaxViT-T ≈ Swin-T.

### H2.6 — Why CNN and ViT recipes differ (to be checked against Blocks 1–2)
CNN: tolerant to optimiser (SGD ≈ AdamW), to warm-up, prefers frozen BN with few samples per batch.
ViT: needs AdamW + warm-up + LLRD; normalisation is per-token (LayerNorm), so the batch
composition (H1.3) does not matter for it at all.

---

## Block 3 — Self-supervised encoders (scripts/ssl_frozen.py, configs/p3/ssl_tune.yaml, scripts/dino_adapt.py)

Encoders: DINOv2-S/14 reg4 (LVD-142M), DINOv3-S/16 (LVD-1689M), MAE ViT-B/16 (IN-1k). Frozen = CLS +
mean patch token per 50 mm tile; sample = mean and std over tiles. Hyper-parameters of k-NN / ridge
are picked by an inner leave-one-site-out CV (nested), never on the outer fold. Written while the
CNN grid was running (Block 1 partially known: ResNet-18 ≈ 38–47).

### H3.1 — Frozen features
**Hypothesis.** DINO-style features are linearly informative about texture statistics (they were
trained to match augmented views, which makes them describe "stuff" well), MAE features are not
linearly separable without fine-tuning (pixel reconstruction objective).
**Prediction.** DINOv2-S ridge: 36–42 (a public notebook reports 34.7 on sample-grouped folds; site
folds cost ~4.5, H0.3). DINOv3-S within ± 3 of DINOv2-S. MAE-B ridge worse than DINOv2-S by ≥ 5.
k-NN is worse than ridge by ≥ 3 for every encoder (with 19 training samples there are no close
neighbours, ridge can interpolate). Tile-level linear probe ≈ sample-level ridge (± 3).

### H3.2 — Can fine-tuning be worse than frozen features?
**Hypothesis.** Yes. Full fine-tuning with a randomly initialised head sends large gradients
through the encoder in the first steps and distorts features that already linearly encode the
target (Kumar et al., LP-FT). With 19 samples the distortion is never repaired.
**Prediction.** DINOv2-S full FT is worse than DINOv2-S frozen ridge by ≥ 2. LP-FT beats full FT by
≥ 2 and lands within ± 2 of frozen. LoRA (r = 8) is within ± 2 of LP-FT at a fraction of the
trainable parameters. CKA (fold 0, 256 validation tiles, per block) between the encoder before and
after tuning: blocks 1–4 > 0.9 for all methods; the last block < 0.6 after full FT and > 0.75 after
LP-FT / LoRA.

### H3.3 — Label-destroying augmentations and domain adaptation
**Hypothesis.** Anything that changes mm/px changes the answer: RandomResizedCrop / zoom, also strong
blur (removes fine grains). Standard DINO multi-crop is built on exactly those (global views 32–100 %
of a region, local 5–32 %, all resized), so continued DINO with default views teaches *scale
invariance*, the opposite of what the task needs.
**Prediction.** Continued DINO for 10 × 100 steps on training tiles: with scale-preserving views
(all crops at 4.5 px/mm, never resized) the frozen-ridge score improves by 1–3 EMD; with standard
RRC multi-crop it does not improve and is ≥ 1 worse than the scale-preserving run. After adaptation,
LP-FT keeps the same ranking. Cost: < 0.5 GPU-hour per adaptation run on the laptop GPU, so the
"helped by how much per GPU-hour" answer is cheap either way.

### H3.4 — SSL vs supervised pretraining
**Prediction.** The best SSL configuration (frozen or LP-FT) is at least as good as the best CNN
(within ± 3) at a fraction of the training cost, and better than the best supervised ViT by ≥ 2.
