# %% [markdown]
# # Parts 1–3 — CNN, ViT and self-supervised results, ablations, training curves, CKA
#
# Everything here is read from `experiments/*/` (written by `scripts/train.py`, `scripts/ssl_frozen.py`,
# `scripts/dino_adapt.py`); predictions and their verdicts are in `HYPOTHESES.md` (Blocks 1–3, S, H2.7, H2.8).
# Site-grouped 5-fold CV EMD, mean ± std over folds; seed-to-seed noise is ≈ 0.2–0.7 (replicates), so
# differences below ~2 are not effects.

# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from soilgs import paths
from soilgs.experiment import results_table

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
res = results_table()
res = res[~res.name.str.startswith(("smoke", "final_", "cka_"))]

def block(prefix, keep=None):
    t = res[res.name.str.startswith(prefix)].sort_values("cv_mean")
    if keep is not None:
        t = t[t.name.str.contains(keep)]
    return t[["name", "cv_mean", "cv_std", "gpu_hours", "peak_mem_gb", "device"]].round(3).reset_index(drop=True)

def history(name):
    return pd.read_csv(paths.EXPERIMENTS / name / "history.csv").groupby("epoch")[["train_loss", "val_emd"]].mean()

# %% [markdown]
# ## Part 1 — CNNs: architectures × optimisation setups (ablation)

# %%
display(block("p1_"))

# %%
ABL = {"optimiser / schedule": ["p1_r18_adamw", "p1_r18_sgd", "p1_r18_adamw_nowarmup", "p1_r18_adamw_lr1e-3", "p1_r18_adamw_nested"],
       "BatchNorm with few samples per batch": ["p1_r18_adamw", "p1_r18_bn_k2", "p1_r18_frozenbn", "p1_r18_frozenbn_k2", "p1_r18_gn"],
       "augmentation": ["p1_r18_noaug", "p1_r18_adamw", "p1_r18_color", "p1_r18_scaleaug", "p1_r18_blur"],
       "input resolution / context": ["p1_r18_res112", "p1_r18_adamw", "p1_r18_tile100", "p1_r18_sgd", "p1_r18_sgd_ctx100"],
       "loss / pretraining": ["p1_r18_adamw", "p1_r18_sqrtmse", "p1_r18_scratch"],
       "architecture (AdamW recipe)": ["p1_r18_adamw", "p1_r50", "p1_effb0", "p1_convnext_t"]}
cv = res.set_index("name")
for title, names in ABL.items():
    names = [n for n in names if n in cv.index]
    print(f"— {title}:  " + " | ".join(f"{n.replace('p1_', '')} {cv.loc[n, 'cv_mean']:.1f}" for n in names))

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
for a, names, t in ((ax[0], ["p1_r18_adamw", "p1_r18_sgd", "p1_r18_scratch"], "ResNet-18: optimiser / pretraining"),
                    (ax[1], ["p2_vits", "p2_vits_sgd", "p2_tinyvit_scratch", "p2_swin_t"], "Transformers"),
                    (ax[2], ["p3_dinov2_s_ft", "p3_dinov2_s_lpft", "p3_dinov3_s_ft", "p3_mae_b_lpft"], "SSL fine-tuning")):
    for i, n in enumerate(names):
        if not (paths.EXPERIMENTS / n / "history.csv").exists():
            continue
        h = history(n)
        a.plot(h.index, h.val_emd, "-", color=f"C{i}", label=f"{n} val")
        a.plot(h.index, h.train_loss, ":", color=f"C{i}")
    a.set(title=t + "\n(solid: validation EMD, dotted: training EMD)", xlabel="epoch", ylabel="EMD", ylim=(0, 90)); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(paths.ROOT / "reports" / "fig_training_curves.png", dpi=110); plt.show()

# %% [markdown]
# ## Part 2 — ViTs: from scratch → pretrained, optimisation ablation

# %%
display(block("p2_"))

# %% [markdown]
# ## Part 3 — Self-supervised encoders: frozen → LP → LP-FT → FT / LoRA, before and after domain adaptation

# %%
rows = []
for enc in ("dinov2_s", "dinov3_s", "mae_b"):
    r = {"encoder": enc}
    for col, name in (("frozen k-NN", f"p3_{enc}_frozen_knn"), ("frozen ridge", f"p3_{enc}_frozen_ridge"),
                      ("frozen ridge 100 mm", f"p3_{enc}_ctx100_frozen_ridge"), ("linear probe", f"p3_{enc}_lp"),
                      ("LP-FT", f"p3_{enc}_lpft"), ("full FT", f"p3_{enc}_ft"), ("LoRA r=8", f"p3_{enc}_lora8"),
                      ("FT 100 mm", f"p3_{enc}_ft_ctx100"), ("LP-FT 100 mm", f"p3_{enc}_lpft_ctx100")):
        r[col] = cv.loc[name, "cv_mean"] if name in cv.index else np.nan
    rows.append(r)
for tag in ("da_fixed", "da_rrc"):
    r = {"encoder": f"dinov2_s after DINO adaptation ({'scale-preserving views' if tag == 'da_fixed' else 'standard multi-crop'})"}
    for col, name in (("frozen k-NN", f"p3_dinov2_s_{tag}_frozen_knn"), ("frozen ridge", f"p3_dinov2_s_{tag}_frozen_ridge"),
                      ("linear probe", f"p3_dinov2_s_{tag}_lp"), ("LP-FT", f"p3_dinov2_s_{tag.replace('da_', 'da_')}_lpft")):
        r[col] = cv.loc[name, "cv_mean"] if name in cv.index else np.nan
    rows.append(r)
display(pd.DataFrame(rows).set_index("encoder").round(1))

# %% [markdown]
# ### Domain adaptation run (continued DINO on unlabelled training tiles): training curves and cost

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for mode in ("fixed_scale", "rrc"):
    d = paths.EXPERIMENTS / f"p3_da_dinov2_s_{mode}"
    h = pd.read_csv(d / "history.csv")
    c = json.loads((d / "compute.json").read_text())
    ax[0].plot(h.epoch, h.dino_loss, "o-", label=f"{mode} ({c['hours'] * 60:.0f} min on {c['device']})")
    m = h.dropna(subset=["frozen_ridge_oof_emd"]) if "frozen_ridge_oof_emd" in h else h.iloc[:0]
    ax[1].plot(m.epoch, m.frozen_ridge_oof_emd, "o-", label=mode)
ax[0].set(title="DINO self-distillation loss", xlabel="epoch"); ax[0].legend(fontsize=8)
ax[1].axhline(cv.loc["p3_dinov2_s_frozen_ridge", "cv_mean"], color="k", ls=":", label="before adaptation (CV)")
ax[1].set(title="frozen-ridge OOF EMD of the teacher (monitor)", xlabel="epoch"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(paths.ROOT / "reports" / "fig_domain_adaptation.png", dpi=110); plt.show()

# %% [markdown]
# ### CKA between the encoder before and after tuning (fold 0, 256 held-out tiles, per block)
# Linear CKA of each transformer block's mean-token output: 1 = same representation up to rotation/scale.

# %%
from soilgs.interpret import load_trained
from soilgs.ssl import cka_by_block, load_encoder
from soilgs.tiles import TARGET_PPM, tile_grid
from soilgs.train import Data, get_device

device = get_device()
data = Data.load()
fold0 = data.folds[data.folds.fold == 0].sample_id
t = int(round(50 * TARGET_PPM))
tiles = [data.images[p][y:y + t, x:x + t] for p in data.photos[data.photos.sample_id.isin(fold0) & (data.photos.split == "train")].photo_id
         for y, x, _ in tile_grid(*data.images[p].shape[:2], 50)]
tiles = np.stack(tiles[:: max(1, len(tiles) // 256)][:256])
base, mean, std = load_encoder("dinov2_s")
cka = {}
for label, name in (("full FT", "cka_dinov2_s_ft"), ("LP-FT", "cka_dinov2_s_lpft"), ("LoRA r=8", "cka_dinov2_s_lora8")):
    if not (paths.ROOT / "checkpoints" / name / "fold0.pt").exists():
        continue
    tuned, _, _, _ = load_trained(name, 0, device)
    cka[label] = cka_by_block(base, tuned.cpu(), tiles, mean, std, device)
    del tuned; torch.cuda.empty_cache()
cka = pd.DataFrame(cka, index=[f"block {i + 1}" for i in range(12)])
display(cka.round(3).T)
cka.plot(marker="o", figsize=(7, 3.2), title="CKA(pretrained DINOv2-S, tuned) per block", ylim=(0, 1.02))
plt.tight_layout(); plt.savefig(paths.ROOT / "reports" / "fig_cka.png", dpi=110); plt.show()
