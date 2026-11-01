"""HS.2: reproduce the public "DINOv2 features + sqrt-mass ridge" pipeline (frozen DINOv2-S CLS features of
five 104 mm crops at 518 px / 5 px/mm per photo, mean + std over crops per sample, ridge on sqrt-mass
targets) and score it on our folds, with nested alpha and with alpha picked on the outer folds.

Difference kept on purpose: crops come from our tray-cropped cache (theirs include the tray frame).

    uv run python scripts/check_public_dinov2.py
"""
import numpy as np
import pandas as pd
import timm
import torch

from soilgs import paths
from soilgs.data import load_labels
from soilgs.experiment import compute_log, save_experiment
from soilgs.folds import make_folds
from soilgs.metric import COLS, emd
from soilgs.ssl import _ridge_fit_predict, frozen_oof
from soilgs.tiles import TARGET_PPM
from soilgs.train import Data, amp_dtype, get_device, resize_to

PUBLIC_PPM, SIZE, LONG_MARGIN, POSITIONS = 5.0, 518, 0.12, (0, .25, .5, .75, 1)


def public_crops(img):
    """Five SIZE x SIZE crops along the long side at 5 px/mm, centred on the short side (public notebook)."""
    f = PUBLIC_PPM / TARGET_PPM
    import cv2
    img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    long_side, short_side = max(h, w), min(h, w)
    margin = max(0, min(int(short_side * LONG_MARGIN), (long_side - SIZE) // 2))
    span, s0 = max(long_side - SIZE - 2 * margin, 0), max((short_side - SIZE) // 2, 0)
    out = []
    for p in POSITIONS:
        l0 = round(margin + p * span)
        out.append(img[s0:s0 + SIZE, l0:l0 + SIZE] if w >= h else img[l0:l0 + SIZE, s0:s0 + SIZE])
    return out


@torch.no_grad()
def features(data):
    device = get_device()
    model = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0).to(device).eval()
    cfg = model.pretrained_cfg
    mean, std = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1), torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    rows = []
    for r in data.photos.itertuples():
        crops = np.stack(public_crops(data.images[r.photo_id])).astype(np.float32) / 255
        x = (torch.from_numpy(crops).permute(0, 3, 1, 2).to(device) - mean) / std
        with torch.autocast(device.type, dtype=amp_dtype(), enabled=device.type == "cuda"):
            f = model(x).float().cpu().numpy()  # pooled [CLS] token, as in the public notebook
        rows += [(r.split, r.sample_id, *v) for v in f]
    df = pd.DataFrame(rows).rename(columns={0: "split", 1: "sample_id"}).set_index(["split", "sample_id"])
    g = df.groupby(level=[0, 1])
    return pd.concat([g.mean(), g.std().fillna(0)], axis=1, keys=["mean", "std"])


def outer_alpha_oof(S, labels, folds, alpha):
    """Optimistic: one alpha for all folds, chosen afterwards on the outer score (public-notebook style)."""
    S = S.loc["train"]
    lab, fm = labels.set_index("sample_id"), folds.set_index("sample_id")
    ids = S.index.tolist()
    X, Y, fold = S.to_numpy(float), lab.loc[ids, COLS].to_numpy(float), fm.loc[ids, "fold"].to_numpy()
    P = np.zeros_like(Y)
    for k in np.unique(fold):
        P[fold == k] = _ridge_fit_predict(X[fold != k], Y[fold != k], X[fold == k], alpha, "sqrt_mass")
    return pd.DataFrame(P, index=ids, columns=COLS)


def main():
    labels, data = load_labels(), Data.load()
    site_folds = pd.read_csv(paths.FOLDS_CSV)
    sample_folds = make_folds(labels, COLS, group="sample", seed=42)
    with compute_log() as c:
        S = features(data)
    cfg = {"pipeline": "public notebook: DINOv2-S/14 reg4 CLS, 5 x 104 mm crops at 518 px (5 px/mm), "
                       "sample mean+std, ridge on sqrt-mass", "crops_from": "tray-cropped cache"}
    y = labels.set_index("sample_id")
    for tag, folds in (("site", site_folds), ("sample", sample_folds)):
        oof, chosen = frozen_oof(S, labels, folds, method="ridge")
        m = save_experiment(f"s_public_dinov2_{tag}_nested", {**cfg, "folds": tag, "alpha": f"nested {chosen}"},
                            oof, labels, folds, c, part="S")
        best = min(((a, emd(y.loc[o.index, COLS].to_numpy(float), o.to_numpy()), o) for a in (0.1, 1, 10, 100, 1000, 10000)
                    for o in [outer_alpha_oof(S, labels, folds, a)]), key=lambda t: t[1])
        m2 = save_experiment(f"s_public_dinov2_{tag}_outer_alpha", {**cfg, "folds": tag, "alpha": f"best on outer folds: {best[0]}"},
                             best[2], labels, folds, c, part="S")
        print(f"{tag:6s} folds: nested alpha CV {m['cv_mean']:.2f} ± {m['cv_std']:.2f} (OOF {m['oof_emd']:.2f}) | "
              f"alpha {best[0]} picked on outer folds: CV {m2['cv_mean']:.2f} (OOF {m2['oof_emd']:.2f})", flush=True)


if __name__ == "__main__":
    main()
