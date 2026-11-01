"""Block C: ResNet-18 SGD with colour normalisation, standard and cross-device validation.

    uv run python scripts/colour_block.py [--only name]
"""
import sys

import yaml

from soilgs.train import run_cross_device_cv, run_cv

spec = yaml.safe_load(open("configs/p1/cnn.yaml"))
base = {**{k: v for k, v in spec.items() if k not in ("experiments", "part")},
        **{k: v for k, v in next(e for e in spec["experiments"] if e["name"] == "p1_r18_sgd").items() if k != "name"}}
RUNS = {"c_r18_sgd_greyworld": ("std", "greyworld"), "c_r18_sgd_grey": ("std", "grey"),
        "c_r18_sgd_xdev": ("xdev", None), "c_r18_sgd_greyworld_xdev": ("xdev", "greyworld"), "c_r18_sgd_grey_xdev": ("xdev", "grey")}
only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(RUNS)
for name in only:
    mode, colour = RUNS[name]
    cfg = {**base, "colour_norm": colour, "num_workers": 3}
    (run_cv if mode == "std" else run_cross_device_cv)(name, cfg, part="C")
