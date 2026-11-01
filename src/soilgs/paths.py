"""Filesystem layout. Override the data root with SOILGS_DATA (e.g. on Colab / Kaggle)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("SOILGS_DATA", ROOT / "data"))
RAW = Path(os.environ.get("SOILGS_RAW", DATA / "raw"))
CACHE = Path(os.environ.get("SOILGS_CACHE", DATA / "cache"))
EXPERIMENTS = ROOT / "experiments"

LABELS_CSV = RAW / "Training_labels_updated.csv"
PPM_CSV = RAW / "ppm_updated.csv"
SAMPLE_SUBMISSION_CSV = RAW / "sample_submission.csv"
TRAIN_DIR = RAW / "Training-All_Photos_updated" / "Training-All_Photos_updated"
TEST_DIR = RAW / "Test_All_Photos" / "Test_All_Photos"

# Small derived artefacts that are versioned in git (folds, photo index).
FOLDS_CSV = DATA / "folds.csv"
PHOTOS_CSV = DATA / "photos.csv"
