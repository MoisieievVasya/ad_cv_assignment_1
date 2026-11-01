"""H3.5: which input factor makes frozen DINOv2-S better — context (mm), pixel density, token readout.

    uv run python scripts/ablate_frozen_input.py
"""
import numpy as np
import pandas as pd

from soilgs import paths
from soilgs.data import load_labels
from soilgs.experiment import compute_log, save_experiment
from soilgs.ssl import frozen_oof, sample_matrix, tile_features
from soilgs.train import Data

# (tile_mm, input_px): 25/50/100 mm at 4.5 px/mm, and 104 mm at 5 px/mm (the public notebook's crop)
SETTINGS = [(25, 112), (50, 224), (100, 448), (104, 518)]


def main():
    labels, folds, data = load_labels(), pd.read_csv(paths.FOLDS_CSV), Data.load()
    rows = []
    for tile_mm, px in SETTINGS:
        with compute_log() as c:
            meta, X = tile_features("dinov2_s", data=data, tile_mm=tile_mm, input_px=px)
        dim = X.shape[1] // 2
        for readout, cols in (("cls_mean", slice(None)), ("cls", slice(0, dim))):
            oof, chosen = frozen_oof(sample_matrix(meta, X[:, cols]), labels, folds, method="ridge")
            name = f"p3_abl_dinov2_s_{tile_mm}mm_{px}px_{readout}"
            m = save_experiment(name, {"encoder": "dinov2_s", "tile_mm": tile_mm, "input_px": px,
                                       "px_per_mm": round(px / tile_mm, 2), "readout": readout,
                                       "tiles": "full grid", "alpha": f"nested {chosen}"},
                                oof, labels, folds, c, part="3")
            rows.append({"tile_mm": tile_mm, "input_px": px, "readout": readout, "cv": m["cv_mean"], "std": m["cv_std"],
                         "tiles_per_photo": round(len(meta) / meta.photo_id.nunique(), 1)})
            print(rows[-1], flush=True)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
