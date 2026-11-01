"""Physically calibrated input pipeline.

1. Every photo is tray-cropped and resampled ONCE to a common px/mm and cached as PNG
   (data/cache/norm_<ppm>/<photo_id>.png). Training photos are already ~4.6 px/mm, so 4.5 px/mm is the
   finest scale shared by every photo; test photos (14-19.5 px/mm) are downsampled to it.
2. Tiles have a fixed size in millimetres; `tile_grid` returns their pixel boxes on a cached photo.
   Training code crops tiles from the cached photos (random positions = augmentation that keeps mm/px).
3. Tile predictions are averaged to a photo, photo predictions to a sample (`aggregate`).
"""
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from . import paths
from .data import crop_tray, load_photo, resample

TARGET_PPM = 4.5


def cache_dir(ppm=TARGET_PPM):
    return paths.CACHE / f"norm_{ppm:g}"


def cached_path(photo_id, ppm=TARGET_PPM):
    return cache_dir(ppm) / f"{photo_id}.png"


def _normalize_one(args):
    path, photo_id, from_ppm, to_ppm = args
    out = cached_path(photo_id, to_ppm)
    img = load_photo(path)
    crop, box = crop_tray(img)
    norm = resample(crop, from_ppm, to_ppm)
    if not out.exists():
        cv2.imwrite(str(out), cv2.cvtColor(norm, cv2.COLOR_RGB2BGR))
    return dict(photo_id=photo_id, crop_y0=box[0], crop_y1=box[1], crop_x0=box[2], crop_x1=box[3],
                norm_h=norm.shape[0], norm_w=norm.shape[1],
                crop_frac=(box[1] - box[0]) * (box[3] - box[2]) / (img.shape[0] * img.shape[1]))


def build_cache(photos, ppm=TARGET_PPM, workers=8):
    """Normalise all photos to `ppm`; returns per-photo crop geometry."""
    cache_dir(ppm).mkdir(parents=True, exist_ok=True)
    jobs = [(r.path, r.photo_id, r.ppm, ppm) for r in photos.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        return pd.DataFrame(list(ex.map(_normalize_one, jobs)))


def load_cached(photo_id, ppm=TARGET_PPM):
    return cv2.cvtColor(cv2.imread(str(cached_path(photo_id, ppm))), cv2.COLOR_BGR2RGB)


def tile_grid(h, w, tile_mm, ppm=TARGET_PPM):
    """Boxes (y, x, size) covering the image with tiles of `tile_mm` mm; overlap only where needed."""
    t = int(round(tile_mm * ppm))
    if h < t or w < t:
        return []
    ys = np.linspace(0, h - t, int(np.ceil(h / t))).round().astype(int)
    xs = np.linspace(0, w - t, int(np.ceil(w / t))).round().astype(int)
    return [(y, x, t) for y in ys for x in xs]


def iter_tiles(img, tile_mm, ppm=TARGET_PPM):
    for y, x, t in tile_grid(*img.shape[:2], tile_mm, ppm):
        yield img[y:y + t, x:x + t]


def aggregate(df, value_cols, levels=("photo_id", "sample_id")):
    """Mean over tiles -> photo, then mean over photos -> sample (each photo weighs the same)."""
    photo = df.groupby(list(levels), as_index=False)[value_cols].mean()
    return photo.groupby(levels[-1])[value_cols].mean()
