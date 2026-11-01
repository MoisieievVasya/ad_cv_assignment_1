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

### Block 2 — results (15 runs, A100, 0.04–0.37 GPU-h each)

CV = mean ± std over the 5 site folds (EMD). References: texture ridge 38.2, ResNet-18 SGD 37.9,
ResNet-18 AdamW 45.7.

| Run | CV | | Run | CV |
|---|---|---|---|---|
| own ViT, scratch, patch 16 | **36.4 ± 14.7** | | ViT-S no LLRD | 46.7 ± 14.6 |
| own ViT, scratch, patch 8 | 39.5 ± 13.6 | | ViT-S drop-path 0.3 | 46.7 ± 17.3 |
| Swin-T (IN-22k) | 44.3 ± 16.7 | | ViT-S colour jitter | 46.9 ± 18.6 |
| MaxViT-T | 44.5 ± 16.1 | | ViT-S no warm-up | 46.9 ± 18.4 |
| ViT-S SGD | 45.1 ± 16.9 | | ViT-S lr 1e-3 | 47.2 ± 16.1 |
| ViT-S 384 px (2.1 mm patches) | 45.7 ± 17.6 | | ViT-S (reference recipe) | 47.6 ± 18.8 |
| DeiT-S (IN-1k) | 48.5 ± 19.2 | | ViT-S wd on all params | 47.6 ± 18.8 |
| ViT-S drop-path 0 | 48.7 ± 18.7 | | | |

| | Prediction | Verdict |
|---|---|---|
| H2.1 | pretrained CNN ≈ pretrained ViT ≫ scratch ViT (60–85) | **Wrong, inverted**: scratch ViT is the best model so far (36.4); pretrained ViT-S 47.6 |
| H2.2 | no warm-up −≥2, lr 1e-3 −≥5, SGD −≥5 | **Wrong**: 46.9 / 47.2 / 45.1 vs 47.6, all within 2.5, SGD even better |
| H2.3 | LLRD helps ≥1.5; wd placement ± 1 | LLRD **wrong** (no-LLRD 46.7); wd placement confirmed (47.6) |
| H2.4 | dp 0 ≈ 0.1; dp 0.3 worse ≥1.5 | dp 0 confirmed (48.7, +1.2); dp 0.3 **wrong** (46.7, better) |
| H2.5 | 384 px better ≥2; DeiT worse ≥2; Swin best transformer, within 2 of best CNN; MaxViT ≈ Swin | 384 px +1.9 (borderline); DeiT −0.9 (**wrong**); Swin best pretrained transformer ✓ but 6.4 behind ResNet-18 SGD ✗; MaxViT ≈ Swin ✓ |

**Why — one mechanism explains most of the misses: the fixed 15-epoch budget overfits every
pretrained transformer.** The per-epoch validation curves (monitoring only, never used to pick a
model) show every pretrained transformer at its best after 1–3 epochs (ViT-S 42.4 at epoch 3,
Swin 42.8 at epoch 2, ViT-S SGD 40.6 at epoch 3) and then drifting up by 4–6 EMD while the training
loss keeps falling (ViT-S 56 → 14). Recipe knobs that shape the *early* optimisation (warm-up, LR,
LLRD, weight-decay placement) are washed out by the end of training. The ±2 differences between
recipes are far below the fold std (15–19) and, presumably, the seed noise (replicates pending on
Kaggle). They should not be read as effects. SGD and strong drop-path "help" only because they slow
down the memorisation.

**Why the scratch ViT wins.** It is small (6 blocks × 192 dims) and starts from nothing, so within
1500 steps it can only learn coarse, sample-level statistics (brightness, colour, amount of
high-frequency texture). Its validation EMD is ~37 from epoch 1 and stays flat (train loss 19.5),
and its per-sample errors correlate 0.86 with the hand-crafted texture ridge: it is effectively a
learned version of the same features. The pretrained ViT has the capacity to memorise each training
sample's appearance. Where that hurts is visible per sample: on fine silty soils it is far worse
than the scratch model (H405 78 vs 25, H666 99 vs 37, H549 35 vs 10, H493 35 vs 10), while on coarse
soils the two are equal. Every model fails on H038 / H037 (well-graded soils whose gravel hides
under a fine surface layer) and H616 (the only clean sand), which is a limit of surface photos, not
of the model.

**Answer to "why does pretraining matter more for ViTs than for CNNs?" — not observed here, and
why.** The textbook argument (ViTs lack locality priors, so they need pretraining) assumes the task
needs object-level features learned from many images. This target is a sample-level texture
statistic, learnable by a tiny model from 19 samples. Pretraining's extra capacity mostly adds a way
to overfit. The from-scratch *ResNet-18* result (Block 1, pending) is the matching control.

**Consequence for later blocks.** Training length is the dominant hyper-parameter for pretrained
models on this dataset. Caveat: an epoch count read off these validation curves would be tuned on
the outer folds (optimistic), so it will be tested as a new hypothesis with nested selection, not
just applied.

### H2.7 — Training length, chosen without peeking (configs/p2/nested_epochs.yaml)
Written after Block 2 and before these runs; Part 3 tuning results were not yet known.
**Hypothesis.** The Block 2 misses come from memorisation late in training. If the stopping epoch is
chosen by an inner CV that never sees the outer fold, pretrained models keep most of their early
(epoch 1–3) quality.
**Prediction.** Chosen epochs are 1–4 for the pretrained models and ≥ 8 for the scratch ViT.
Nested early stopping improves ViT-S from 47.6 to ≤ 43, Swin-T from 44.3 to ≤ 41, ResNet-18 AdamW from
45.7 to ≤ 41, and DINOv2-S full fine-tuning likewise by ≥ 4. The scratch ViT changes by < 1.5.
Even so, no pretrained model beats the scratch ViT by more than 2. The early-epoch curves in
Block 2 bottomed out at ~41–42, and nested selection is noisier than reading the minimum off the
outer curve.

---

## Sanity block S — are we doing something wrong? (written before both runs)

Context: every model family stops at CV ≈ 36–38 (best: scratch ViT 36.4, ResNet-18 SGD 37.9,
texture ridge 38.2; ResNet-50 45.0). A classmate reports "mean validation EMD 17" with ResNet-50, and a
public notebook reports frozen DINOv2-S + ridge = 34.7 where ours gives 49.0.

### HS.1 — A leaky split explains numbers like 17
**Hypothesis.** Each sample has 3–8 photos of the same tray. If photos, not samples, are split between
train and validation, a network that can memorise a sample's appearance is validated on soils it was
trained on. Our ResNet-50 already reaches EMD 16 on its *training* samples at epoch 6.
**Prediction.** The exact ResNet-18 SGD recipe (CV 37.9 with site-grouped folds), re-run with a
random 5-fold split of *photos*, scores ≤ 22 EMD (held-out photos aggregated per sample), i.e. at least
15 points "better" than the honest number. The texture ridge, which cannot memorise, moved only from
36.9 to 34.2 under the same leak (measured while writing this), so the gap is specific to
high-capacity models.

### HS.2 — Is our 50 mm / 224 px tiling too small for frozen pretrained ViTs?
**Hypothesis (two competing explanations of 34.7 vs 49.0).**
(a) Context: the public notebook feeds DINOv2 five 104 mm crops at 518 px (≈ 5 px/mm), so the
model sees whole gravel clasts and the tray-scale arrangement; our 50 mm tiles at 224 px may be too
small for features learned on object-centric images.
(b) Validation: their folds are sample-grouped (not site), the fold seed was searched over 5000 seeds
for balance, and ridge α was chosen on the same folds that are reported.
**Prediction.** Reproducing their pipeline (5 × 104 mm crops at 518 px, CLS features, mean + std over
crops per sample, ridge on √mass) on *our* site folds with *nested* α gives 42–46. That is better
than our 49.0, but far from 34.7, so (b) explains more of the gap than (a). Their α picked on the outer
folds is ≥ 2 more optimistic than nested α. If the context effect is ≥ 5 EMD, the ViT/SSL blocks are
re-run with 100 mm tiles.

---

## Block 6a — data efficiency (configs/p6/data_efficiency.yaml), written before the runs

Best recipe per family so far: ResNet-18 SGD (37.9), own scratch ViT (36.4), DINOv3-S fine-tuned (42.2).
**Hypothesis.** Pretraining is a data prior: the less labelled data, the more it should matter. The
scratch ViT has no prior at all, so it should degrade fastest when training samples are removed.
**Prediction.** With 50 % of each fold's training samples (≈ 9–10 samples): scratch ViT +6 to +10 EMD,
ResNet-18 SGD +4 to +8, DINOv3-S FT +2 to +5. At 50 % the ranking flips and DINOv3-S FT is the best
of the three. At 75 % all three move by less than half of their 50 % change.

### Sanity block S — results

| | Prediction | Result | Verdict |
|---|---|---|---|
| HS.1 leaky photo split, ResNet-18 SGD | ≤ 22, ≥ 15 better than honest | **25.2** vs 36.5 honest (OOF): 11 better | **Direction confirmed, size over-predicted** |
| HS.2 public DINOv2 pipeline on our site folds | 42–46; validation (b) explains more than context (a); outer-α ≥ 2 optimistic | **40.5** (nested α) vs our 49.0; on sample folds 35.2 (they report 34.7); outer α 40.3 | **Wrong**: context/input (a) is worth ≈ 8.5 EMD, site grouping ≈ 5, α choice ≈ 0.2 |

**HS.1 — why.** With other photos of the same tray in training, the network recognises the tray:
samples it can memorise drop to 4–8 EMD (H371 14.8 → 3.8, H374 39.8 → 7.6, H493 23.3 → 4.2), while
soils that differ between photos (H126, H030) do not improve. A reported "validation EMD 17"
with ResNet-50 (more capacity than ResNet-18, and our ResNet-50 reaches 16 on its *training*
samples) is consistent with such a split plus picking the best epoch on the validation set. Our
site-grouped CV stays the only number we use.

**HS.2 — why it was wrong, and what it changes.** The public pipeline differs from ours in three
ways at once: 104 mm instead of 50 mm context, 518 px at 5 px/mm instead of 224 px at 4.5 px/mm,
and the [CLS] token only (we used [CLS] + mean patch token). Together they are worth ~8.5 EMD for frozen
DINOv2. Validation choices explain the rest of the gap to 34.7 (site vs sample folds ≈ 5,
outer-fold α ≈ 0.2, fold-seed search: not measured). Which of the three input factors matters is
tested next (H3.5), before any Block 2/3 model is re-run with larger crops.

### H3.5 — Which input factor makes frozen DINOv2 8.5 EMD better? (written before the ablation)
Frozen DINOv2-S, nested ridge, site folds; one factor changed at a time, all crops on our grid of
the tray-cropped cache unless stated.
**Hypothesis.** Context dominates: a 50 mm crop shows at most a few gravel clasts and no
arrangement, while the encoder was trained on object-centric scenes where 100 mm of soil looks like
"a surface". Pixel density (4.5 vs 5 px/mm) and the token readout ([CLS] vs [CLS] + mean patch) are
second-order. Five fixed crops vs the full grid should not matter (both average over the photo).
**Prediction.** 100 mm at 4.5 px/mm (448 px) improves on 50 mm (224 px) by ≥ 5. [CLS]-only vs
[CLS] + mean differs by ≤ 2 at either size. 104 mm at 5 px/mm (518 px) on the full grid is within
± 1.5 of the 5-crop replication (40.5). 25 mm (112 px) is worse than 50 mm by ≥ 3.

### H3.5 — results (scripts/ablate_frozen_input.py; frozen DINOv2-S, nested ridge, site folds)

| crop | input | px/mm | tiles/photo | [CLS] + mean patch | [CLS] only |
|---|---|---|---|---|---|
| 25 mm | 112 px | 4.5 | 95 | 49.0 | 52.1 |
| 50 mm | 224 px | 4.5 | 26 | 49.0 | 50.5 |
| **100 mm** | 448 px | 4.5 | 9 | **42.3** | 45.2 |
| 104 mm | 518 px | 5.0 | 6 | 44.3 | 44.5 |
| 104 mm, 5 fixed central crops (public pipeline, HS.2) | 518 px | 5.0 | 5 | — | 40.5 |

| | Prediction | Verdict |
|---|---|---|
| 100 mm better than 50 mm by ≥ 5 | −6.7 | **Confirmed**: context is the main factor |
| [CLS] only vs [CLS] + mean within 2 | mean patch helps by 0.3–3.1 | Mostly confirmed (3.1 at 25 mm and 2.9 at 100 mm exceed it) |
| 104 mm full grid within ± 1.5 of 5 central crops | 44.3–44.5 vs 40.5 | **Wrong** |
| 25 mm worse than 50 mm by ≥ 3 | equal (49.0) | **Wrong**: below 50 mm nothing changes, above it a lot does |

**Why.** A 50 mm crop at 4.5 px/mm shows the encoder a nearly uniform texture patch, where the
object-centric pretrained features have little to work with. At 100 mm whole gravel clasts and their
spacing are visible, which these features do describe. The readout question matches the
texture-statistic view: the mean patch token, which weighs every 3 mm region equally, adds information
over [CLS].
**Grid vs five central crops (−4).** Restricting the 104 mm grid to tiles centred on the tray's
short axis gives 40.1, edge rows alone give 47.4: the tray edges (shadow of the frame, soil
disturbed while filling the tray) carry less useful signal. At 100 mm the same split shows
no difference (42.2 vs 42.8), so this position effect is not robust. It is tested again with
trained models in Block 4 (centre vs border tiles).
**Decision.** Frozen SSL encoders are re-scored at 100 mm / 448 px. Whether fine-tuned ViTs / SSL
models also gain from 100 mm context is a new hypothesis (training at 448 px costs ~4×).

### H2.8 — Does 100 mm context help *trained* models too? (configs/ctx100.yaml, written before the runs)
Known when writing: frozen DINOv2 gains 6.7 from 100 mm context (H3.5); a ResNet-18 *AdamW* at
100 mm / 450 px was not better than at 50 mm (46.5 vs 45.7, Block 1); MAE-B LP-FT at 50 mm = 40.95.
**Hypothesis.** The gain is specific to pretrained transformers: their features describe objects and
arrangements, which need context. The scratch ViT and the CNN learn local texture statistics,
which a 50 mm tile already contains (cf. H0.4).
**Prediction.** Pretrained ViT-S, DINOv3-S FT and MAE-B LP-FT each improve by ≥ 3 EMD at 100 mm;
ResNet-18 SGD and the scratch ViT change by < 2. MAE-B LP-FT at 100 mm becomes the best single model
(≤ 37). Cost ≈ 3–4× the 50 mm runs.

**H3.5 follow-up — the other frozen encoders at 100 mm / 448 px** (nested ridge, site folds):
DINOv2-S 49.0 → **42.3**, DINOv3-S 45.8 → 46.5, MAE-B 45.1 → 44.3. The context gain is
DINOv2-specific, not a general property of pretrained transformers. A plausible reason is the
positional embeddings: DINOv2 was pretrained at 518 px (37 × 37 patches) and is interpolated *down*
to 16 × 16 for 224 px inputs, while DINOv3 and MAE were trained at 224 px. Then 448 px input moves
DINOv2 closer to its training resolution but moves the others away from theirs. Not tested further.
This weakens H2.8's prediction for DINOv3 and MAE, which was written before these numbers.

---

## Block 4 — interpretation and shortcut controls (written before the runs)

Models: ResNet-18 SGD (best CNN), own scratch ViT (best overall), pretrained ViT-S (contrast), frozen
DINOv2-S (SSL; attention rollout of the pretrained encoder). All maps and controls are on held-out
fold models.
**H4.1 — what the maps show.** Grad-CAM of predicted coarseness (CNN) lights up individual
gravel clasts and their shadows on coarse soils and is diffuse on fine soils. Attention rollout of
the pretrained ViTs concentrates on a few high-contrast spots (bright clasts, specular particles),
not uniformly on the texture.
**H4.2 — colour shortcut (main control, numeric).** Clay/silt-rich soils are darker and more
uniform, so colour alone predicts a lot. *Prediction:* a ridge on six Lab colour statistics per
tile scores 45–55 (vs mean curve 93). Feeding the trained models tiles replaced by their mean
colour raises the scratch ViT's EMD by ≤ 10, but the ResNet-18's by ≥ 15. The scratch ViT leans
on colour more. Greyscale input costs the scratch ViT ≥ 5 and the ResNet-18 ≤ 3.
**H4.3 — texture vs layout.** Shuffling 4 × 4 patches inside a tile (12.5 mm pieces) costs ≤ 3
for every model (they read texture statistics, not layout); 8 × 8 (6 mm pieces, cutting gravel)
costs ≥ 5 on gravelly samples.
**H4.4 — tray position.** Predicting only from tiles near the tray border is ≥ 3 worse than only
from central tiles, for every model.
**H4.5 — device shortcut.** A "model" that predicts the mean curve of the training samples
photographed with the same phone is no better than the plain mean curve (≥ 85).

## Block 6b — robustness to scale (written before the runs)
**Prediction.** (a) Same photo at lower resolution with correct px/mm (×0.75, ×0.5, ×0.35 of
4.5 px/mm, then resampled back): every model degrades monotonically. At ×0.5 the ResNet-18 loses
≤ 5, the scratch ViT ≤ 5, the pretrained ViT-S ≥ 5. (b) Wrong px/mm by ×1.25 / ×0.8: every model
loses 5–12 EMD (a pure shift costs 9.7, H0.1, but models are less sensitive than the metric).

## Block 7 — ensemble (written before the runs)
**Prediction.** Per-sample OOF errors correlate ≥ 0.8 between all trained models (the hard samples
H037/H038/H616 dominate), and lowest between a trained model and frozen DINOv2 at 100 mm (≤ 0.7).
An equal-weight average of 3–4 models from different families beats the best single model by
2–4 EMD. Adding the best-but-similar models (two ResNet-18 recipes) gains < 1.

### H2.7 — results (nested early stopping; inner 3-fold site CV picks the stop epoch per outer fold)

| Model | 15 epochs | nested stop | chosen epochs (5 folds) | Prediction | Verdict |
|---|---|---|---|---|---|
| ViT-S (IN-21k) | 47.6 | **39.6** | 8, 1, 1, 1, 4 | ≤ 43 | **Confirmed** (−8.0) |
| DINOv2-S full FT | 45.5 | 42.4 | 1, 1, 2, 3, 8 | ≥ 4 better | Partly (−3.1) |
| ResNet-18 AdamW | 45.7 | 43.9 | 4, 10, 2, 5, 5 | ≤ 41 | **Wrong** (−1.8) |
| Swin-T | 44.3 | 46.7 | 3, 5, 2, 5, 7 | ≤ 41 | **Wrong** (+2.4) |
| own scratch ViT | 36.4 | 36.8 | 5, 7, 2, 4, 4 | change < 1.5, epochs ≥ 8 | change confirmed, epochs **wrong** |

No pretrained model beats the scratch ViT (prediction confirmed). Seed replicates of the reference
recipes (repeats_top) put the run-to-run noise at ≈ 0.2–0.7 EMD std, and fold std is ≈ 15. So
ViT-S's −8 is real, while DINOv2 −3.1, ResNet-18 −1.8 and Swin +2.4 are at the edge of the noise.
**Why.** Plain ViT-S memorises fastest (Block 2: best epoch 3), so stopping early helps it most.
The nested choice is itself noisy: an inner CV on ~19 samples / ~10 sites picks epochs between 1
and 10 for the same model. Where the outer curve is flat (ResNet-18, Swin), that noise cancels
the benefit. The scratch ViT does not overfit, so it does not care where it stops. Conclusion for the
CNN/ViT recipe comparison: training length matters more than any other optimisation knob for
pretrained *ViTs* on this dataset. For CNNs the SGD optimiser does a similar job (slower fitting:
ResNet-18 SGD 37.9 without early stopping).

### Block 1 — results (19 CNN runs + nested / 100 mm follow-ups; noise floor: seed std ≈ 0.2–0.7, fold std ≈ 15)

| | Prediction | Result | Verdict |
|---|---|---|---|
| H1.1 | ResNet-18 AdamW 34–42; R50 ≈ R18 (± 2); ConvNeXt-T best CNN by < 4; EffNet-B0 ≈ R18 | R18 AdamW 45.7, R50 45.0, ConvNeXt-T 46.5, EffNet-B0 49.9; best CNN = **R18 SGD 37.9** | R50 ≈ R18 ✓; rest **wrong**: bigger/better-ImageNet nets do not help |
| H1.2 | SGD vs AdamW within 2; no warm-up within 1; lr 1e-3 ≥ 2 worse | SGD **37.9** vs AdamW 45.7; no warm-up 44.3; lr 1e-3 47.1 | **Wrong** for SGD (−7.8); warm-up/LR within noise |
| H1.3 | BN with 2-sample batches ≥ 3 worse; frozen BN closes the gap; GN 1–3 worse than frozen BN | BN-k2 46.8 (+1.1), frozen BN 44.0, frozen BN-k2 39.7, **GN 38.4** | **Wrong**: no batch-composition penalty beyond noise; GN is the best BN variant |
| H1.4 | no aug ≥ 1 worse; colour neutral; scale aug ≥ 5 worse; blur ≥ 2 worse | no aug 43.3, colour 44.0, scale aug 42.4, blur 42.4 (base 45.7) | colour ✓; the "label-destroying" augs did **not** hurt (**wrong**) |
| H1.5 | 112 px ≥ 3 worse; 100 mm within ± 2 | 112 px 41.7 (better), 100 mm 46.5 | 100 mm ✓, resolution **wrong** |
| H1.6 | EMD loss ≥ 2 better than sqrt-MSE | 45.7 vs 45.9 | **Wrong** (no difference) |
| H1.7 | R18 from scratch 50–75 | **39.3**, better than ImageNet R18 with AdamW | **Wrong** |

**Why — one picture behind almost all misses.** The pretrained CNN with AdamW memorises the 19
training samples within ~3 epochs (train EMD → 10–13, validation EMD rises from its minimum ~39 to
~46; ResNet-50: 16 on training samples at epoch 6). From there, *anything that slows memorisation helps*.
That covers SGD (37.9), swapping BN for GroupNorm (38.4; the BN-calibrated pretrained weights are partly
de-calibrated), starting from scratch (39.3), lower resolution (41.7), and the "label-destroying"
scale/blur augmentations (42.4: they inject noise the network cannot memorise). Bigger ImageNet models
(R50, ConvNeXt, EffNet) memorise *more*. The predictions assumed that the networks *measure grain
size* and are limited by information. In practice they are limited by overfitting, and grain-size
precision is not what they exploit: if it were, ×[0.5, 2] scale noise would cost ~20 EMD (Block 0),
not −3.
**Inductive biases of CNNs on soil texture.** *Helpful:* locality and weight sharing (a stationary
texture statistic can be learned from 19 samples; ResNet-18 *from scratch* reaches 39.3). Global
average pooling also helps: it averages a texture statistic over the tile, matching how the label is
defined. *Harmful here:* ImageNet object features plus capacity, which make it easy to recognise the
individual tray instead of its grain size.
**BatchNorm with tiles of few samples.** Batches drawn from only 2 samples did not hurt measurably
(46.8 vs 45.7 with train-mode BN; 39.7 vs 44.0 with frozen BN, inside the ±15 fold noise). With
19 samples × many tiles, BN's statistics in a 2-sample batch are biased towards those trays: the
model learns slightly batch-dependent features, but evaluation uses running statistics averaged over
all batches. The effect is real in principle and too small to detect against the overfitting effect.
GroupNorm removes the question entirely and was the best normalisation (38.4).

### Block 3 — results (frozen / LP / LP-FT / FT / LoRA; before and after DINO domain adaptation)

| | frozen k-NN | frozen ridge | ridge 100 mm | linear probe | LP-FT | full FT | LoRA r=8 |
|---|---|---|---|---|---|---|---|
| DINOv2-S | 52.4 | 49.0 | **42.3** | 61.0 | 43.1 | 45.5 (nested 42.4) | 46.6 |
| DINOv3-S | 51.6 | 45.8 | 46.5 | 56.3 | 43.2 | **42.2** | 45.6 |
| MAE-B | 43.6 | 45.1 | 44.3 | 62.7 | **41.0** | 46.9 | — |
| DINOv2-S after DA, scale-preserving views | 47.0 | 50.0 | — | 65.4 | 47.3 | — | — |
| DINOv2-S after DA, standard multi-crop | 46.9 | 48.9 | — | 63.0 | 46.6 | — | — |

| | Prediction | Verdict |
|---|---|---|
| H3.1 | DINOv2 ridge 36–42; DINOv3 within ± 3; MAE ≥ 5 worse; k-NN ≥ 3 worse than ridge; LP ≈ ridge | DINOv2 49.0 at 50 mm (**wrong**), 42.3 at 100 mm ✓; DINOv3 −3.2 (borderline); MAE **better** (45.1, **wrong**); k-NN worse for DINOv2/v3 ✓, better for MAE ✗; tile-level LP 56–63 (**wrong**) |
| H3.2 | FT ≥ 2 worse than frozen; LP-FT ≥ 2 better than FT ≈ frozen; LoRA ≈ LP-FT | MAE: FT 46.9 worse than frozen 45.1 ✓, LP-FT 41.0 (−5.9) ✓; DINOv2: LP-FT −2.4 vs FT ✓, but FT better than 50 mm-frozen; DINOv3: FT ≈ LP-FT; LoRA 2.4–3.5 worse than LP-FT ✗ |
| H3.3 | DA with scale-preserving views −1…−3; standard multi-crop no gain and ≥ 1 worse than scale-preserving | **Both hurt**: frozen ridge 49.0 → 50.0 (scale-preserving) / 48.9 (multi-crop); LP-FT 43.1 → 47.3 / 46.6; cost 0.11 / 0.13 L4-hours ✓ | **Wrong** |
| H3.4 | best SSL within ± 3 of best CNN; ≥ 2 better than best supervised ViT | MAE LP-FT 41.0 vs R18 SGD 37.9 (3.1); vs ViT-S nested 39.6 (worse) | **Wrong** (borderline) |

**Why.** (1) *Frozen features* need context (H3.5: 100 mm, DINOv2-specific). MAE features are
pixel-level and, contrary to the prediction, they are already reasonable for a texture target.
(2) *Can fine-tuning be worse than frozen?* Yes, for MAE: full FT 46.9 vs frozen 45.1. And
DINOv2 frozen at 100 mm (42.3) beats every DINOv2 fine-tuning at 50 mm. LP-FT is the safest tuning
recipe (best for MAE and DINOv2): starting the encoder update from a fitted head limits the
feature distortion (Kumar et al.). This was the mechanism predicted (CKA in
`notebooks/01_results_parts_1_3.ipynb`). The tile-level linear probe is poor because it treats ~4000
correlated tiles as independent examples with 19 distinct labels and overfits them. The sample-level
ridge does not have that problem. (3) *Label-destroying augmentations* (by construction): random
resized crops, zoom, strong blur — anything that changes mm/px or removes fine grains. (4)
*Domain adaptation failed for both view policies.* With 127 photos of 24 trays, continued DINO
quickly solves its pretext task (loss 8.2 → 3.6 in 10 epochs). The easiest cue that "two views come
from the same photo" is the tray's colour, illumination and moisture pattern, not grain size. The
monitor shows frozen-ridge OOF EMD rising from 45.3 / 44.5 to 48.4 / 47.2 as soon as the backbone
starts training. Scale-preserving views make that shortcut *easier* (same scale, same tray), which
is why they hurt slightly more than standard multi-crop. Verdict: DA made every score worse by 1–4 EMD
at ~0.1 L4-hours per run.

### H1.8 / H7.2 — combining findings for the final models (written before these runs)
**H1.8.** SGD (37.9) and GroupNorm (38.4) both help ResNet-18 by slowing memorisation of the training
samples; they act on the same mechanism, so they are not additive. *Prediction:* ResNet-18 SGD + GN lands
within ± 2 of 37.9.
**H7.2.** Averaging three seeds of the same model removes only the seed variance (std 0.3–0.7).
*Prediction:* the seed-averaged OOF of ResNet-18 SGD and of the scratch ViT is 0.3–1.5 better than the
mean of their single seeds, which is far less than averaging across families (H7: 2–4).

**H3.2 — CKA result** (fold 0, 256 held-out tiles, linear CKA of each block's mean token, DINOv2-S
before vs after tuning; `notebooks/01_results_parts_1_3.ipynb`, `reports/fig_cka.png`):

| | block 1 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| full FT | 1.00 | 0.99 | 0.94 | 0.83 | 0.61 | **0.43** |
| LP-FT | 1.00 | 1.00 | 0.98 | 0.92 | 0.85 | **0.75** |
| LoRA r=8 | 1.00 | 0.94 | 0.77 | 0.65 | 0.54 | **0.38** |

Predictions: blocks 1–4 > 0.9 for all ✓; last block < 0.6 after full FT ✓ (0.43); > 0.75 after LP-FT
(0.745, borderline ✓) and after LoRA ✗ (0.38). **Why LoRA moved the representation most:** it was
trained with a 10× larger LR (2e-4, the usual LoRA setting) and updates q/k/v in *every* block, so
attention patterns change throughout the depth even with only 0.15 M trainable parameters. LP-FT,
starting from a fitted head with a 1e-5 LR, keeps the representation closest to the pretrained one,
and it is the best DINOv2 tuning (43.1). That is the LP-FT argument: the less the features are
distorted, the better on 19 samples.
