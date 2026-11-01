"""Photo index, physical scale and image loading.

Scale facts this module encodes:
- `ppm_updated.csv` gives px/mm at each phone's *native* resolution.
- The organisers downscaled most training photos (Motorola Edge 4000 -> 1600 px, Samsung 9248 -> 1599 px),
  so the effective px/mm is native_ppm * long_side / native_long_side (~4.6 px/mm for those photos).
- Test photos are native (14-19.5 px/mm) and stored as MPO; frame 0 is the photo.
"""
import re
import unicodedata

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from . import paths
from .metric import COLS

_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"})


def normalize_key(text):
    """Lower-case ASCII alphanumerics only; makes file names and sample ids comparable."""
    text = unicodedata.normalize("NFKD", str(text).translate(_UMLAUTS)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def load_labels():
    lab = pd.read_csv(paths.LABELS_CSV)
    lab.columns = ["sample_id"] + COLS
    lab["sample_id"] = lab["sample_id"].astype(str)
    return lab


def load_cameras():
    cams = pd.read_csv(paths.PPM_CSV)
    cams["key"] = cams["phone"].map(normalize_key)
    cams["native_long"] = cams[["width", "height"]].max(axis=1)
    return cams


def build_photo_index():
    """One row per photo: split, sample_id, camera, native and effective px/mm, size."""
    lab = load_labels()
    sub = pd.read_csv(paths.SAMPLE_SUBMISSION_CSV)
    cams = load_cameras().sort_values("key", key=lambda s: -s.str.len())  # longest prefix first
    rows = []
    for split, folder, ids in (("train", paths.TRAIN_DIR, lab.sample_id), ("test", paths.TEST_DIR, sub.sample_id.astype(str))):
        id_keys = {sid: normalize_key(sid) for sid in ids}
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            key = normalize_key(path.stem)
            matches = [sid for sid, k in id_keys.items() if k in key]
            cam = next((c for c in cams.itertuples() if key.startswith(c.key)), None)
            if len(matches) != 1 or cam is None:
                raise ValueError(f"cannot resolve sample/camera for {path.name}: {matches}, {cam}")
            with Image.open(path) as im:
                w, h = im.size
            shot = int(re.findall(r"(\d+)\D*$", path.stem)[0])
            rows.append(dict(split=split, sample_id=matches[0], camera=cam.phone, shot=shot, path=str(path),
                             width=w, height=h, native_ppm=cam.ppm,
                             ppm=cam.ppm * max(w, h) / cam.native_long))
    df = pd.DataFrame(rows)
    df["photo_id"] = (df.split + "__" + df.sample_id.map(normalize_key) + "__" + df.camera.map(normalize_key)
                      + "__" + df.groupby(["split", "sample_id", "camera"]).cumcount().astype(str))
    return df


def load_photos():
    """Versioned photo index (data/photos.csv) with absolute paths for this machine."""
    photos = pd.read_csv(paths.PHOTOS_CSV)
    photos["path"] = photos.path.map(lambda p: str(paths.RAW / p))
    return photos


def load_photo(path):
    """RGB uint8 array, EXIF orientation applied, first frame for MPO files."""
    with Image.open(path) as im:
        im.seek(0)
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def crop_tray(img, occupancy=(0.6, 0.45, 0.3), dark=0.6, inset=0.03, min_frac=0.55):
    """Crop away the dark tray frame, the LED strip and the background.

    Pixels brighter than `dark` x the 75th percentile of lightness (and not blown out) count as 'soil';
    the crop grows from the centre while rows/columns stay mostly soil. Rows through the frame are almost
    entirely dark, rows through cobbles only partly: the strictest occupancy that still keeps
    >= `min_frac` of the photo is used (cobble photos need a looser one). Two earlier
    references failed: colour similarity to the centre (public notebook "soil-psd-physics-not-deep-learning")
    and the centre's median lightness, because many trays have a darker, damp patch in the middle.
    Returns the crop and its box (y0, y1, x0, x1).
    """
    H, W = img.shape[:2]
    s = 400 / max(H, W)
    small = cv2.GaussianBlur(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (0, 0), 2)
    L = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    h, w = L.shape
    soil = (L > dark * np.percentile(L, 75)) & (L < 250)

    def grow(profile, c, occ):
        a = b = c
        while a > 0 and profile[a - 1] > occ:
            a -= 1
        while b < len(profile) - 1 and profile[b + 1] > occ:
            b += 1
        return a, b + 1

    for occ in occupancy:
        y0, y1 = grow(soil.mean(1), h // 2, occ)
        x0, x1 = grow(soil.mean(0), w // 2, occ)
        if (y1 - y0) * (x1 - x0) >= min_frac * h * w:
            break
    y0, y1, x0, x1 = int(y0 / s), int(y1 / s), int(x0 / s), int(x1 / s)
    dy, dx = int(inset * (y1 - y0)), int(inset * (x1 - x0))
    box = (y0 + dy, y1 - dy, x0 + dx, x1 - dx)
    return img[box[0]:box[1], box[2]:box[3]], box


def resample(img, from_ppm, to_ppm):
    """Change px/mm. INTER_AREA when shrinking (anti-aliasing), bicubic when enlarging."""
    f = to_ppm / from_ppm
    if abs(f - 1) < 1e-3:
        return img
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


def normalized_photo(row, target_ppm):
    """Tray-cropped photo resampled to `target_ppm` px/mm."""
    crop, _ = crop_tray(load_photo(row.path))
    return resample(crop, row.ppm, target_ppm)
