# Results table

CV = mean ± std of the competition EMD over the 5 site-grouped folds (`data/folds.csv`, same for every
experiment); OOF = EMD of the pooled out-of-fold sample curves. Lower is better. Public / private LB only
for experiments that were submitted (`reports/leaderboard.csv`). Seed noise ≈ 0.2–0.7 (replicates `rep_*`).
Mean-curve baseline 93.4, uniform curve 102.

## Part 0 — baselines

| experiment | CV | OOF | public LB | private LB | GPU-h | device |
|---|---|---|---|---|---|---|
| p0_texture_ridge_100mm | 38.15 ± 14.86 | 36.64 |  |  | 0.000 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p0_texture_ridge_50mm | 38.17 ± 14.62 | 36.91 |  |  | 0.000 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p0_texture_ridge_25mm | 38.55 ± 13.28 | 37.75 |  |  | 0.000 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p0_mean_curve | 93.41 ± 9.58 | 94.68 |  |  | 0.000 | NVIDIA GeForce RTX 5070 Laptop GPU |

## Part 1 — CNN

| experiment | CV | OOF | public LB | private LB | GPU-h | device |
|---|---|---|---|---|---|---|
| p1_r18_sgd_gn | 36.39 ± 13.35 | 35.64 |  |  | 0.169 | NVIDIA L4 |
| p1_r18_sgd | 37.91 ± 15.09 | 36.48 |  |  | 0.094 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_r18_gn | 38.41 ± 14.42 | 36.84 |  |  | 0.121 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_r18_scratch | 39.32 ± 14.55 | 38.21 |  |  | 0.178 | NVIDIA L4 |
| p1_r18_frozenbn_k2 | 39.71 ± 15.35 | 38.18 |  |  | 0.090 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_r18_res112 | 41.73 ± 15.73 | 39.58 |  |  | 0.061 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_r18_scaleaug | 42.37 ± 13.11 | 40.54 |  |  | 0.095 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_r18_blur | 42.41 ± 14.18 | 40.53 |  |  | 0.109 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_r18_noaug | 43.28 ± 15.28 | 40.88 |  |  | 0.097 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_r18_adamw_nested | 43.88 ± 15.25 | 41.55 |  |  | 0.433 | NVIDIA A100-SXM4-40GB |
| p1_r18_color | 43.96 ± 15.38 | 41.46 |  |  | 0.096 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_r18_frozenbn | 44.04 ± 14.80 | 41.87 |  |  | 0.091 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_r18_adamw_nowarmup | 44.33 ± 15.48 | 41.86 |  |  | 0.094 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_r50 | 44.96 ± 13.35 | 43.15 |  |  | 0.465 | NVIDIA L4 |
| p1_r18_adamw | 45.72 ± 17.09 | 42.83 |  |  | 0.099 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_r18_sqrtmse | 45.91 ± 15.79 | 43.72 |  |  | 0.172 | NVIDIA L4 |
| p1_r18_tile100 | 46.47 ± 18.47 | 42.77 |  |  | 0.276 | NVIDIA L4 |
| p1_convnext_t | 46.51 ± 17.08 | 43.48 |  |  | 0.445 | NVIDIA L4 |
| p1_r18_bn_k2 | 46.79 ± 13.25 | 48.09 |  |  | 0.096 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_r18_adamw_lr1e-3 | 47.14 ± 14.36 | 44.74 |  |  | 0.094 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p1_effb0 | 49.88 ± 18.89 | 46.69 |  |  | 0.379 | NVIDIA L4 |

## Part 2 — ViT

| experiment | CV | OOF | public LB | private LB | GPU-h | device |
|---|---|---|---|---|---|---|
| p2_tinyvit_scratch | 36.41 ± 14.74 | 34.89 |  |  | 0.087 | NVIDIA A100-SXM4-40GB |
| p2_tinyvit_scratch_nested | 36.79 ± 15.17 | 36.66 |  |  | 0.544 | NVIDIA A100-SXM4-40GB |
| p2_tinyvit_scratch_p8 | 39.46 ± 13.61 | 38.50 |  |  | 0.088 | NVIDIA A100-SXM4-40GB |
| p2_vits_nested | 39.59 ± 14.93 | 39.19 |  |  | 0.697 | NVIDIA A100-SXM4-40GB |
| p2_swin_t | 44.35 ± 16.66 | 41.48 |  |  | 0.264 | NVIDIA A100-SXM4-40GB |
| p2_maxvit_t | 44.52 ± 16.13 | 41.66 |  |  | 0.375 | NVIDIA A100-SXM4-40GB |
| p2_vits_sgd | 45.08 ± 16.91 | 42.77 |  |  | 0.203 | NVIDIA A100-SXM4-40GB |
| p2_vits_384 | 45.66 ± 17.63 | 42.95 |  |  | 0.277 | NVIDIA A100-SXM4-40GB |
| p2_vits_nollrd | 46.68 ± 14.65 | 44.10 |  |  | 0.201 | NVIDIA A100-SXM4-40GB |
| p2_vits_dp03 | 46.71 ± 17.32 | 44.28 |  |  | 0.209 | NVIDIA A100-SXM4-40GB |
| p2_swin_t_nested | 46.72 ± 16.43 | 44.42 |  |  | 0.921 | NVIDIA A100-SXM4-40GB |
| p2_vits_color | 46.90 ± 18.58 | 43.84 |  |  | 0.233 | NVIDIA A100-SXM4-40GB |
| p2_vits_nowarmup | 46.91 ± 18.38 | 43.59 |  |  | 0.130 | NVIDIA A100-SXM4-40GB |
| p2_vits_lr1e-3 | 47.22 ± 16.14 | 44.47 |  |  | 0.208 | NVIDIA A100-SXM4-40GB |
| p2_vits | 47.57 ± 18.79 | 44.17 |  |  | 0.131 | NVIDIA A100-SXM4-40GB |
| p2_vits_wd_all | 47.63 ± 18.78 | 44.24 |  |  | 0.204 | NVIDIA A100-SXM4-40GB |
| p2_deit_s | 48.48 ± 19.18 | 45.28 |  |  | 0.191 | NVIDIA A100-SXM4-40GB |
| p2_vits_dp0 | 48.73 ± 18.69 | 45.43 |  |  | 0.202 | NVIDIA A100-SXM4-40GB |

## Part 3 — self-supervised

| experiment | CV | OOF | public LB | private LB | GPU-h | device |
|---|---|---|---|---|---|---|
| p3_mae_b_lpft | 40.95 ± 15.31 | 39.68 |  |  | 0.882 | NVIDIA L4 |
| p3_dinov3_s_ft | 42.21 ± 16.97 | 41.19 |  |  | 0.490 | NVIDIA L4 |
| p3_dinov2_s_ctx100_frozen_ridge | 42.27 ± 15.93 | 40.69 |  |  | 0.000 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_abl_dinov2_s_100mm_448px_cls_mean | 42.27 ± 15.93 | 40.69 |  |  | 0.002 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_dinov2_s_ft_nested | 42.39 ± 17.65 | 40.79 |  |  | 0.772 | NVIDIA A100-SXM4-40GB |
| p3_dinov2_s_lpft | 43.12 ± 16.21 | 41.64 |  |  | 0.523 | NVIDIA L4 |
| p3_dinov3_s_lpft | 43.22 ± 17.17 | 41.94 |  |  | 0.548 | NVIDIA L4 |
| p3_mae_b_frozen_knn | 43.57 ± 19.84 | 40.75 |  |  | 0.000 | NVIDIA L4 |
| p3_abl_dinov2_s_104mm_518px_cls_mean | 44.28 ± 18.08 | 42.38 |  |  | 0.002 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_mae_b_ctx100_frozen_ridge | 44.34 ± 17.90 | 42.02 |  |  | 0.000 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_abl_dinov2_s_104mm_518px_cls | 44.55 ± 16.94 | 42.91 |  |  | 0.002 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_mae_b_ctx100_frozen_knn | 44.71 ± 15.00 | 42.72 |  |  | 0.000 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_mae_b_frozen_ridge | 45.11 ± 18.48 | 42.67 |  |  | 0.002 | NVIDIA L4 |
| p3_abl_dinov2_s_100mm_448px_cls | 45.22 ± 16.91 | 43.14 |  |  | 0.002 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_dinov2_s_ft | 45.47 ± 17.55 | 43.27 |  |  | 0.492 | NVIDIA L4 |
| p3_dinov3_s_ctx100_frozen_knn | 45.55 ± 16.05 | 42.66 |  |  | 0.000 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_dinov3_s_lora8 | 45.64 ± 17.45 | 43.67 |  |  | 0.281 | NVIDIA L4 |
| p3_dinov3_s_frozen_ridge | 45.82 ± 15.97 | 43.86 |  |  | 0.001 | NVIDIA L4 |
| p3_dinov3_s_ctx100_frozen_ridge | 46.50 ± 17.16 | 44.22 |  |  | 0.000 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_dinov2_s_da_rrc_lpft | 46.56 ± 17.11 | 43.86 |  |  | 0.357 | NVIDIA L4 |
| p3_dinov2_s_lora8 | 46.59 ± 18.82 | 43.77 |  |  | 0.280 | NVIDIA L4 |
| p3_mae_b_ft | 46.87 ± 17.73 | 44.92 |  |  | 0.870 | NVIDIA L4 |
| p3_dinov2_s_da_rrc_frozen_knn | 46.93 ± 12.04 | 45.30 |  |  | 0.000 | NVIDIA L4 |
| p3_dinov2_s_da_fixed_frozen_knn | 47.04 ± 13.22 | 45.76 |  |  | 0.000 | NVIDIA L4 |
| p3_dinov2_s_da_fixed_lpft | 47.30 ± 15.92 | 45.08 |  |  | 0.358 | NVIDIA L4 |
| p3_dinov2_s_da_rrc_frozen_ridge | 48.92 ± 17.10 | 47.19 |  |  | 0.001 | NVIDIA L4 |
| p3_abl_dinov2_s_25mm_112px_cls_mean | 48.97 ± 18.97 | 46.07 |  |  | 0.001 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_abl_dinov2_s_50mm_224px_cls_mean | 49.00 ± 18.48 | 46.13 |  |  | 0.001 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_dinov2_s_frozen_ridge | 49.03 ± 18.49 | 46.16 |  |  | 0.001 | NVIDIA L4 |
| p3_dinov2_s_da_fixed_frozen_ridge | 50.03 ± 16.75 | 48.43 |  |  | 0.001 | NVIDIA L4 |
| p3_abl_dinov2_s_50mm_224px_cls | 50.47 ± 17.60 | 48.21 |  |  | 0.001 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_dinov2_s_ctx100_frozen_knn | 50.90 ± 19.16 | 48.84 |  |  | 0.000 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_dinov3_s_frozen_knn | 51.63 ± 20.43 | 50.11 |  |  | 0.000 | NVIDIA L4 |
| p3_dinov3_s_ctx100_lp | 51.71 ± 13.13 | 51.59 |  |  | 0.001 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_abl_dinov2_s_25mm_112px_cls | 52.12 ± 18.76 | 49.76 |  |  | 0.001 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_dinov2_s_frozen_knn | 52.41 ± 18.62 | 50.76 |  |  | 0.000 | NVIDIA L4 |
| p3_dinov2_s_ctx100_lp | 52.70 ± 13.31 | 54.65 |  |  | 0.001 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_dinov3_s_lp | 56.33 ± 12.91 | 57.76 |  |  | 0.004 | NVIDIA L4 |
| p3_dinov2_s_lp | 60.99 ± 11.17 | 61.78 |  |  | 0.004 | NVIDIA L4 |
| p3_mae_b_lp | 62.70 ± 12.08 | 63.46 |  |  | 0.004 | NVIDIA L4 |
| p3_dinov2_s_da_rrc_lp | 63.00 ± 13.43 | 64.42 |  |  | 0.004 | NVIDIA L4 |
| p3_mae_b_ctx100_lp | 63.02 ± 6.08 | 63.03 |  |  | 0.001 | NVIDIA GeForce RTX 5070 Laptop GPU |
| p3_dinov2_s_da_fixed_lp | 65.38 ± 10.49 | 66.21 |  |  | 0.004 | NVIDIA L4 |

## Part 6 — data efficiency

| experiment | CV | OOF | public LB | private LB | GPU-h | device |
|---|---|---|---|---|---|---|
| p6_r18_sgd_frac50 | 40.66 ± 16.17 | 39.97 |  |  | 0.157 | NVIDIA L4 |
| p6_tinyvit_frac50 | 45.34 ± 22.42 | 43.73 |  |  | 0.124 | NVIDIA L4 |
| p6_r18_sgd_frac75 | 45.37 ± 16.95 | 43.60 |  |  | 0.158 | NVIDIA L4 |
| p6_tinyvit_frac75 | 45.88 ± 17.43 | 43.74 |  |  | 0.126 | NVIDIA L4 |
| p6_dinov3_s_ft_frac50 | 48.61 ± 19.57 | 49.27 |  |  | 0.336 | NVIDIA L4 |
| p6_dinov3_s_ft_frac75 | 48.63 ± 19.92 | 47.00 |  |  | 0.336 | NVIDIA L4 |

## Sanity checks

| experiment | CV | OOF | public LB | private LB | GPU-h | device |
|---|---|---|---|---|---|---|
| s_leaky_photo_split_r18_sgd | 26.48 ± 17.01 | 25.15 |  |  | 0.075 | NVIDIA GeForce RTX 5070 Laptop GPU |
| s_public_dinov2_sample_outer_alpha | 35.02 ± 7.72 | 34.50 |  |  | 0.002 | NVIDIA GeForce RTX 5070 Laptop GPU |
| s_public_dinov2_sample_nested | 35.24 ± 7.51 | 34.72 |  |  | 0.002 | NVIDIA GeForce RTX 5070 Laptop GPU |
| s_public_dinov2_site_outer_alpha | 40.30 ± 18.12 | 39.47 |  |  | 0.002 | NVIDIA GeForce RTX 5070 Laptop GPU |
| s_public_dinov2_site_nested | 40.50 ± 18.95 | 39.67 |  |  | 0.002 | NVIDIA GeForce RTX 5070 Laptop GPU |

## Final models (fold checkpoints + test)

| experiment | CV | OOF | public LB | private LB | GPU-h | device |
|---|---|---|---|---|---|---|
| final_r18_sgd | 36.51 ± 14.88 | 35.01 |  |  | 0.269 | NVIDIA GeForce RTX 5070 Laptop GPU |
| final_tinyvit_scratch | 36.55 ± 14.77 | 35.20 |  |  | 0.142 | NVIDIA GeForce RTX 5070 Laptop GPU |
| final_tinyvit_scratch_s1 | 37.10 ± 14.65 | 35.71 |  |  | 0.162 | NVIDIA L4 |
| final_tinyvit_scratch_s2 | 37.14 ± 14.73 | 35.74 |  |  | 0.161 | NVIDIA L4 |
| final_r18_sgd_s2 | 37.46 ± 14.84 | 36.19 |  |  | 0.202 | NVIDIA L4 |
| final_r18_sgd_s1 | 37.57 ± 14.16 | 36.23 |  |  | 0.203 | NVIDIA L4 |
| final_mae_b_lpft | 40.82 ± 15.53 | 39.44 |  |  | 0.254 | NVIDIA A100-SXM4-40GB |
| final_vits_nested | 43.42 ± 15.07 | 42.22 |  |  | 0.477 | NVIDIA A100-SXM4-40GB |
| final_vits | 47.77 ± 19.10 | 44.42 |  |  | 0.454 | NVIDIA GeForce RTX 5070 Laptop GPU |

## Seed replicates (noise floor)

| experiment | CV | OOF | public LB | private LB | GPU-h | device |
|---|---|---|---|---|---|---|
| rep_p2_tinyvit_scratch_s1 | 37.55 ± 15.31 | 36.09 |  |  | 0.158 | NVIDIA L4 |
| rep_p2_tinyvit_scratch_s2 | 38.16 ± 14.96 | 36.78 |  |  | 0.157 | NVIDIA L4 |
| rep_p3_dinov3_s_ft_s1 | 41.74 ± 17.23 | 41.12 |  |  | 0.371 | NVIDIA L4 |
| rep_p3_dinov3_s_ft_s2 | 41.74 ± 17.12 | 41.11 |  |  | 0.371 | NVIDIA L4 |

## Other submissions

| submission | public LB | private LB |
|---|---|---|
| final_ensemble_guard.csv (CV 36.39 ± 15.54, `notebooks/06_comparison_ensemble.ipynb`) | 56.39 | revealed after 2026-11-30 |
