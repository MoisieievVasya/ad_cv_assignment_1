"""Hand-crafted texture features in physical units (used as a non-deep baseline and for controls).

The grey image is divided by a large blur (removes illumination gradients), then a stack of
Gaussian blurs gives difference-of-Gaussians (DoG) energy per size band. Bands are in millimetres
when `ppm` is the image's px/mm, or in pixels when `ppm=1`.
"""
import cv2
import numpy as np

BANDS_MM = (0.25, 0.5, 1, 2, 4, 8, 16, 32)
BANDS_PX = (1, 2, 4, 8, 16, 32, 64, 128)


def gray_normalized(img, illum_sigma):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return g / (cv2.GaussianBlur(g, (0, 0), illum_sigma) + 1e-3)


def dog_energy(gn, bands, ppm):
    """RMS of the difference between consecutive blurs, one value per band (band unit * ppm = px)."""
    prev = cv2.GaussianBlur(gn, (0, 0), max(bands[0] / 2 * ppm, 0.5))
    energy = []
    for b in bands:
        cur = cv2.GaussianBlur(gn, (0, 0), b * ppm)
        energy.append(float(np.sqrt(np.mean((prev - cur) ** 2))))
        prev = cur
    return np.array(energy)


def dog_centroid(energy, bands):
    """Energy-weighted mean of log2(band): the dominant texture scale."""
    return float((energy * np.log2(bands)).sum() / energy.sum())


def texture_features(img, ppm):
    """mm-indexed DoG energies, their fractions and centroid, plus Lab colour moments (23 values)."""
    gn = gray_normalized(img, 15 * ppm)
    e = dog_energy(gn, BANDS_MM, ppm)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return np.concatenate([np.log(e + 1e-6), e / e.sum(), [dog_centroid(e, BANDS_MM)],
                           lab.mean(0), lab.std(0)])


FEATURE_NAMES = ([f"log_dog_{b}mm" for b in BANDS_MM] + [f"frac_dog_{b}mm" for b in BANDS_MM]
                 + ["dog_centroid"] + [f"{c}_mean" for c in "Lab"] + [f"{c}_std" for c in "Lab"])


def pixel_centroid(img):
    """Pixel-indexed texture scale (ignores px/mm on purpose) — used to show the resolution gap."""
    gn = gray_normalized(img, 0.05 * max(img.shape[:2]))
    return dog_centroid(dog_energy(gn, BANDS_PX, 1.0), BANDS_PX)
