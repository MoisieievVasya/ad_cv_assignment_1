"""Inference: photo(s) of ONE soil sample + their scale -> a valid grain-size curve.

    uv run python scripts/predict.py photo1.jpg photo2.jpg --ppm 19.525
    uv run python scripts/predict.py tray.jpg --ppm 13.942 --models final_r18_sgd final_tinyvit_scratch --plot curve.png

--ppm is the pixels-per-millimetre of the photos as stored (native phone resolution: see ppm_updated.csv;
if a photo was resized, scale it accordingly). Each photo is tray-cropped, resampled to 4.5 px/mm, cut into
the model's tiles; bin probabilities are averaged over tiles, photos and models; their cumulative sum is
the curve (monotone, in [0, 100], 100 % at 200 mm by construction). Models are the `full.pt` checkpoints
written by `scripts/train.py --test --save-checkpoints`.
"""
import argparse

import numpy as np
import pandas as pd

from soilgs.data import crop_tray, load_photo, resample
from soilgs.interpret import load_trained, photo_probs
from soilgs.metric import COLS, d_percentile
from soilgs.tiles import TARGET_PPM
from soilgs.train import get_device

DEFAULT_MODELS = ["final_r18_sgd", "final_tinyvit_scratch"]


def predict_curve(photo_paths, ppm, models=DEFAULT_MODELS):
    device = get_device()
    images = [resample(crop_tray(load_photo(p))[0], ppm, TARGET_PPM) for p in photo_paths]
    probs = []
    for name in models:
        model, cfg, mean, std = load_trained(name, "full", device)
        probs.append(np.mean([photo_probs(model, im, cfg, mean, std, device) for im in images], 0))
    curve = np.cumsum(np.mean(probs, 0)) * 100
    curve[-1] = 100.0
    return pd.Series(curve, index=COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+", help="photos of one sample")
    ap.add_argument("--ppm", type=float, required=True, help="pixels per mm of the photos as given")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--out", help="write the curve to this CSV")
    ap.add_argument("--plot", help="save a plot of the curve to this PNG")
    args = ap.parse_args()

    curve = predict_curve(args.photos, args.ppm, args.models)
    print(curve.round(2).to_frame("% passing").T.to_string())
    print("D10 / D50 / D60 (mm):", [round(d_percentile(curve.to_numpy(), p), 4) for p in (10, 50, 60)])
    if args.out:
        curve.to_frame("pct_passing").rename_axis("sieve_mm").to_csv(args.out)
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3.5))
        plt.semilogx([float(c) for c in COLS], curve, "o-")
        plt.xlabel("sieve (mm)"); plt.ylabel("% passing"); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(args.plot, dpi=120)


if __name__ == "__main__":
    main()
