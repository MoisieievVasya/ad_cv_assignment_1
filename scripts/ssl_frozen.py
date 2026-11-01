"""Part 3, frozen encoders: k-NN, ridge (nested hyper-parameter choice) and a tile-level linear probe.

    uv run python scripts/ssl_frozen.py                         # all encoders, ImageNet/LVD weights
    uv run python scripts/ssl_frozen.py --encoders dinov2_s --weights checkpoints/dino_adapt_dinov2_s_fixed_scale.pt --tag da_fixed
"""
import argparse

from soilgs.data import load_labels
from soilgs.experiment import compute_log, save_experiment
from soilgs.ssl import ENCODERS, frozen_oof, frozen_test, linear_probe_oof, sample_matrix, tile_features
from soilgs.submission import write_submission
from soilgs import paths
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    ap.add_argument("--weights", help="adapted encoder weights (one encoder only)")
    ap.add_argument("--tag", default="", help="suffix for experiment names, e.g. da_fixed")
    ap.add_argument("--submit", action="store_true", help="write a submission for the ridge model")
    args = ap.parse_args()
    labels, folds = load_labels(), pd.read_csv(paths.FOLDS_CSV)
    for enc in args.encoders:
        tag = f"{enc}_{args.tag}" if args.tag else enc
        with compute_log() as c_feat:
            meta, X = tile_features(enc, weights=args.weights, tag=tag)
        S = sample_matrix(meta, X)
        base = {"encoder": enc, "weights": args.weights or "pretrained", "tile_mm": 50, "input_px": 224,
                "features": "CLS + mean patch token; sample = mean and std over tiles"}
        for method in ("knn", "ridge"):
            with compute_log() as c:
                oof, chosen = frozen_oof(S, labels, folds, method=method)
            m = save_experiment(f"p3_{tag}_frozen_{method}", {**base, "method": method, "chosen_per_fold": chosen},
                                oof, labels, folds, {**c, "feature_hours": c_feat["hours"]}, part="3")
            print(f"[p3_{tag}_frozen_{method}] CV {m['cv_mean']:.2f} ± {m['cv_std']:.2f}  chosen {chosen}", flush=True)
            if args.submit and method == "ridge":
                h = max(set(chosen), key=chosen.count)
                write_submission(frozen_test(S, labels, "ridge", h), paths.ROOT / "submissions" / f"p3_{tag}_frozen_ridge.csv")
        with compute_log() as c:
            oof = linear_probe_oof(meta, X, labels, folds)
        m = save_experiment(f"p3_{tag}_lp", {**base, "method": "linear probe (tile-level softmax head, EMD loss)",
                                             "epochs": 200, "lr": 1e-2, "wd": 1e-3}, oof, labels, folds, c, part="3")
        print(f"[p3_{tag}_lp] CV {m['cv_mean']:.2f} ± {m['cv_std']:.2f}", flush=True)


if __name__ == "__main__":
    main()
