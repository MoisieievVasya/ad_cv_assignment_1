"""Part 3, domain adaptation: continue DINO self-distillation of an encoder on unlabelled training tiles.

    uv run python scripts/dino_adapt.py --mode fixed_scale      # scale-preserving views (all at 4.5 px/mm)
    uv run python scripts/dino_adapt.py --mode rrc              # standard DINO multi-crop (changes mm/px)

Only training photos are used (the competition rules on using unlabelled test photos were not
verified, so test photos are never seen). Every `--monitor-every` epochs the teacher's frozen features
are scored with the nested ridge on the shared folds: that is the training curve of the adaptation.
Writes experiments/p3_da_<encoder>_<mode>/history.csv and checkpoints/dino_adapt_<encoder>_<mode>.pt.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from soilgs import paths
from soilgs.data import load_labels
from soilgs.experiment import compute_log
from soilgs.metric import COLS, emd
from soilgs.ssl import dino_adapt, encode_tiles, frozen_oof, sample_matrix, tile_batches
from soilgs.train import Data, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov2_s")
    ap.add_argument("--mode", choices=["fixed_scale", "rrc"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--monitor-every", type=int, default=2)
    args = ap.parse_args()

    labels, folds = load_labels(), pd.read_csv(paths.FOLDS_CSV)
    data = Data.load(splits=("train",))
    epoch = {"n": 0}

    def monitor(teacher, mean, std):
        epoch["n"] += 1
        if epoch["n"] % args.monitor_every and epoch["n"] != args.epochs:
            return {}
        teacher.eval()
        meta, feats = [], []
        for rows, tiles in tile_batches(data):
            meta += rows
            feats.append(encode_tiles(teacher, tiles, mean, std, get_device()))
        S = sample_matrix(pd.DataFrame(meta), np.concatenate(feats))
        oof, _ = frozen_oof(S, labels, folds, method="ridge")
        teacher.train()
        return {"frozen_ridge_oof_emd": emd(labels.set_index("sample_id").loc[oof.index, COLS].to_numpy(float), oof.to_numpy())}

    name = f"p3_da_{args.encoder}_{args.mode}"
    with compute_log() as c:
        hist, ckpt = dino_adapt(args.encoder, args.mode, epochs=args.epochs, steps_per_epoch=args.steps,
                                batch_size=args.batch, lr=args.lr, monitor=monitor, out_name=f"dino_adapt_{args.encoder}_{args.mode}")
    out = paths.EXPERIMENTS / name
    out.mkdir(parents=True, exist_ok=True)
    hist.to_csv(out / "history.csv", index=False)
    (out / "compute.json").write_text(json.dumps({**c, "checkpoint": str(ckpt.relative_to(paths.ROOT)), **vars(args)}, indent=2))
    print(hist.to_string(index=False))


if __name__ == "__main__":
    main()
