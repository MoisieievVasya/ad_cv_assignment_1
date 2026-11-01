"""Build the photo index, the shared folds file and the normalised-photo cache.

    uv run python scripts/prepare_data.py            # index + folds + cache at 4.5 px/mm
"""
import argparse

from soilgs import paths
from soilgs.data import build_photo_index, load_labels
from soilgs.folds import make_folds
from soilgs.metric import COLS
from soilgs.tiles import TARGET_PPM, build_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppm", type=float, default=TARGET_PPM)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    photos = build_photo_index()
    geometry = build_cache(photos, args.ppm, args.workers)
    photos = photos.merge(geometry, on="photo_id")
    photos["path"] = photos.path.map(lambda p: str(p).replace(str(paths.RAW) + "/", ""))  # relative to RAW
    photos.to_csv(paths.PHOTOS_CSV, index=False)
    print(photos.groupby(["split", "camera"]).agg(n=("photo_id", "size"), ppm=("ppm", "mean"),
                                                 crop_frac=("crop_frac", "mean")).round(3))

    if not paths.FOLDS_CSV.exists():  # folds are created once and then frozen in git
        folds = make_folds(load_labels(), COLS)
        folds.to_csv(paths.FOLDS_CSV, index=False)
        print(folds.groupby("fold").agg(samples=("sample_id", list), sites=("site", "nunique")))


if __name__ == "__main__":
    main()
