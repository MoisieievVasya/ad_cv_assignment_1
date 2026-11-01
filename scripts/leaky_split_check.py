"""HS.1: the ResNet-18 SGD recipe of Block 1 validated with a LEAKY random split over photos.

    uv run python scripts/leaky_split_check.py
"""
import yaml

from soilgs.train import run_photo_split_cv

spec = yaml.safe_load(open("configs/p1/cnn.yaml"))
shared = {k: v for k, v in spec.items() if k not in ("experiments", "part")}
recipe = next(e for e in spec["experiments"] if e["name"] == "p1_r18_sgd")
cfg = {**shared, **{k: v for k, v in recipe.items() if k != "name"}}
run_photo_split_cv("s_leaky_photo_split_r18_sgd", cfg, part="S")
