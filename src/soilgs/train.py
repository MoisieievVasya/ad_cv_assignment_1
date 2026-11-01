"""Tile-level training with sample-level validation, shared by CNN, ViT and SSL fine-tuning runs.

Data: cached photos at 4.5 px/mm (`tiles.py`). A training item is a random `tile_mm` crop of a random photo
of a random training sample (samples are equally likely, so heavily photographed samples don't dominate),
labelled with the sample's curve. Validation tiles every photo of the held-out samples on a fixed grid,
averages bin probabilities tile -> photo -> sample, and scores the sample curve with the competition EMD.

Loss: the model outputs 11 logits; softmax = mass per bin, cumsum = curve, loss = EMD of that curve
(the metric itself, differentiable). Alternative `loss: sqrt_mse` regresses sqrt-mass targets instead.
"""
import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from . import paths
from .data import load_labels, load_photos
from .experiment import compute_log, save_experiment
from .metric import COLS, WEIGHTS
from .models import build_model, freeze_bn_stats, param_groups
from .tiles import TARGET_PPM, load_cached, tile_grid

DEFAULTS = dict(
    arch="resnet18", pretrained=True, norm="bn", drop_path=0.0,
    tile_mm=50, input_px=None,                       # input_px: resize tiles (same for train/test)
    epochs=15, steps_per_epoch=100, batch_size=32, samples_per_batch=None,
    optimizer="adamw", lr=3e-4, weight_decay=0.05, momentum=0.9, warmup_epochs=1, schedule="cosine",
    llrd=1.0, head_lr_mult=1.0, wd_on_all=False, grad_clip=1.0,
    aug_flip=True, aug_rot90=True, aug_color=0.0, aug_scale=None, aug_blur=0.0,
    loss="emd", tta=True, seed=0, num_workers=4, amp=True, freeze_backbone=False,
)


def merge_config(cfg):
    out = copy.deepcopy(DEFAULTS)
    out.update(cfg or {})
    out["img_size"] = out["input_px"] or int(round(out["tile_mm"] * TARGET_PPM))
    return out


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Data:
    labels: pd.DataFrame
    photos: pd.DataFrame
    folds: pd.DataFrame
    images: dict
    mass: dict  # sample_id -> mass per bin (11,) summing to 1

    @classmethod
    def load(cls, splits=("train", "test")):
        labels, photos = load_labels(), load_photos()
        photos = photos[photos.split.isin(splits)]
        folds = pd.read_csv(paths.FOLDS_CSV)
        images = {pid: load_cached(pid) for pid in photos.photo_id}
        curves = labels.set_index("sample_id")[COLS].to_numpy(float)
        mass = np.diff(np.column_stack([np.zeros(len(curves)), curves]), axis=1).clip(0) / 100
        mass = dict(zip(labels.sample_id, mass / mass.sum(1, keepdims=True)))
        return cls(labels, photos, folds, images, mass)


def augment(tile, cfg, rng):
    """Label-preserving: flips, 90° rotations, mild colour jitter. Label-destroying (ablations only):
    random rescale (`aug_scale`, changes mm/px) and blur (`aug_blur`, erases fine grains)."""
    if cfg["aug_flip"] and rng.random() < .5:
        tile = tile[:, ::-1]
    if cfg["aug_flip"] and rng.random() < .5:
        tile = tile[::-1]
    if cfg["aug_rot90"]:
        tile = np.rot90(tile, rng.integers(4))
    tile = np.ascontiguousarray(tile).astype(np.float32) / 255
    if cfg["aug_color"]:
        a = cfg["aug_color"]
        tile = tile * rng.uniform(1 - a, 1 + a) + rng.uniform(-a, a) * .5           # contrast / brightness
        grey = tile.mean(2, keepdims=True)
        tile = grey + (tile - grey) * rng.uniform(1 - a, 1 + a)                       # saturation
    if cfg["aug_blur"] and rng.random() < .5:
        import cv2
        tile = cv2.GaussianBlur(tile, (0, 0), rng.uniform(.5, cfg["aug_blur"]))
    return np.clip(tile, 0, 1)


class TileTrainSet(Dataset):
    def __init__(self, data, sample_ids, cfg, mean, std):
        self.d, self.cfg, self.mean, self.std = data, cfg, np.array(mean, np.float32), np.array(std, np.float32)
        self.samples = list(sample_ids)
        ph = data.photos[data.photos.sample_id.isin(self.samples) & (data.photos.split == "train")]
        self.photos_of = ph.groupby("sample_id").photo_id.apply(list).to_dict()
        self.length = cfg["steps_per_epoch"] * cfg["batch_size"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # idx is either an int (random sample) or (sample index, seed) from GroupedBatchSampler
        rng = np.random.default_rng(None)
        s = self.samples[idx[0]] if isinstance(idx, tuple) else self.samples[rng.integers(len(self.samples))]
        img = self.d.images[self.photos_of[s][rng.integers(len(self.photos_of[s]))]]
        t = int(round(self.cfg["tile_mm"] * TARGET_PPM))
        if self.cfg["aug_scale"]:  # crop a larger/smaller physical area and resize to t: changes mm/px
            t = int(round(t * rng.uniform(*self.cfg["aug_scale"])))
            t = min(t, *img.shape[:2])
        y, x = rng.integers(img.shape[0] - t + 1), rng.integers(img.shape[1] - t + 1)
        tile = augment(img[y:y + t, x:x + t], self.cfg, rng)
        tile = resize_to(tile, self.cfg["img_size"])
        tile = (tile - self.mean) / self.std
        return torch.from_numpy(tile.transpose(2, 0, 1).copy()), torch.from_numpy(self.d.mass[s].astype(np.float32))


def resize_to(tile, size):
    if tile.shape[0] == size:
        return tile
    import cv2
    return cv2.resize(tile, (size, size), interpolation=cv2.INTER_AREA if tile.shape[0] > size else cv2.INTER_CUBIC)


class GroupedBatchSampler(Sampler):
    """Every batch holds tiles of only `k` samples — to study BatchNorm statistics that come from few
    physical samples (the realistic case when photos are tiled)."""

    def __init__(self, n_samples, batch_size, steps, k):
        self.n, self.bs, self.steps, self.k = n_samples, batch_size, steps, k

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            chosen = np.random.choice(self.n, self.k, replace=False)
            yield [(int(chosen[i % self.k]), i) for i in range(self.bs)]


def emd_loss(logits, mass):
    p = logits.float().softmax(1)
    cdf_err = (p.cumsum(1) - mass.cumsum(1)).abs()[:, :-1] * 100
    return (cdf_err * torch.as_tensor(WEIGHTS, device=p.device, dtype=torch.float32)).sum(1).mean()


def sqrt_mse_loss(logits, mass):
    return F.mse_loss(logits.float().softmax(1).clamp_min(0).sqrt(), mass.sqrt())


@torch.no_grad()
def predict_samples(model, data, sample_ids, split, cfg, mean, std, device, batch_size=64):
    """Sample-level curves (DataFrame sample_id x COLS) from grid tiles of every photo."""
    model.eval()
    mean_t, std_t = torch.tensor(mean, device=device).view(1, 3, 1, 1), torch.tensor(std, device=device).view(1, 3, 1, 1)
    t = int(round(cfg["tile_mm"] * TARGET_PPM))
    out = {}
    for s in sample_ids:
        photo_probs = []
        for pid in data.photos[(data.photos.sample_id == s) & (data.photos.split == split)].photo_id:
            img = data.images[pid]
            tiles = np.stack([resize_to(img[y:y + t, x:x + t].astype(np.float32) / 255, cfg["img_size"])
                              for y, x, _ in tile_grid(*img.shape[:2], cfg["tile_mm"])])
            probs = []
            for i in range(0, len(tiles), batch_size):
                xb = (torch.from_numpy(tiles[i:i + batch_size]).permute(0, 3, 1, 2).to(device) - mean_t) / std_t
                with torch.autocast(device.type, dtype=amp_dtype(), enabled=cfg["amp"] and device.type == "cuda"):
                    views = [xb, xb.flip(3), xb.flip(2), xb.transpose(2, 3)] if cfg["tta"] else [xb]
                    p = torch.stack([model(v).float().softmax(1) for v in views]).mean(0)
                probs.append(p.cpu().numpy())
            photo_probs.append(np.concatenate(probs).mean(0))
        out[s] = np.cumsum(np.mean(photo_probs, 0)) * 100
    df = pd.DataFrame(out, index=COLS).T
    df["200"] = 100.0
    return df


def amp_dtype():
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def lr_factor(step, total, warmup, schedule):
    if step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def train_one(data, train_ids, cfg, device, val_ids=None, log_every=1):
    """Train on `train_ids`; returns (model, history DataFrame). History has train loss per epoch and,
    if `val_ids` is given, the validation EMD per epoch (monitoring only, never used to pick epochs)."""
    model = build_model(cfg).to(device)
    data_cfg = model.pretrained_cfg if hasattr(model, "pretrained_cfg") and cfg.get("pretrained", True) else {}
    mean, std = data_cfg.get("mean", (0.485, 0.456, 0.406)), data_cfg.get("std", (0.229, 0.224, 0.225))
    if cfg["freeze_backbone"]:
        for n, p in model.named_parameters():
            p.requires_grad = n.split(".")[0] in ("head", "fc", "classifier")
    groups = param_groups(model, cfg)
    if cfg["optimizer"] == "sgd":
        opt = torch.optim.SGD(groups, lr=cfg["lr"], momentum=cfg["momentum"], nesterov=True)
    else:
        opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.999))
    ds = TileTrainSet(data, train_ids, cfg, mean, std)
    if cfg["samples_per_batch"]:
        loader = DataLoader(ds, batch_sampler=GroupedBatchSampler(len(ds.samples), cfg["batch_size"], cfg["steps_per_epoch"],
                                                                  cfg["samples_per_batch"]),
                            num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0)
    else:
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                            num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0)
    loss_fn = emd_loss if cfg["loss"] == "emd" else sqrt_mse_loss
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype() == torch.float16)
    total, warmup = cfg["epochs"] * cfg["steps_per_epoch"], cfg["warmup_epochs"] * cfg["steps_per_epoch"]
    history, step = [], 0
    for epoch in range(cfg["epochs"]):
        model.train()
        if cfg["norm"] == "frozen_bn":
            freeze_bn_stats(model)
        losses = []
        for xb, yb in loader:
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * g["lr_scale"] * lr_factor(step, total, warmup, cfg["schedule"])
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype(), enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        if val_ids is not None and ((epoch + 1) % log_every == 0 or epoch + 1 == cfg["epochs"]):
            pred = predict_samples(model, data, val_ids, "train", {**cfg, "tta": False}, mean, std, device)
            truth = data.labels.set_index("sample_id").loc[pred.index, COLS].to_numpy(float)
            row["val_emd"] = float((np.abs(truth - pred[COLS].to_numpy())[:, :-1] * WEIGHTS).sum(1).mean())
        history.append(row)
        if not math.isfinite(row["train_loss"]):
            break
    model.mean_std = (mean, std)
    return model, pd.DataFrame(history)


def run_cv(name, cfg, part, folds_to_run=None, predict_test=False, save_checkpoints=False, verbose=True):
    """Full group-CV experiment: OOF curves, per-fold scores, training curves, compute log -> experiments/<name>."""
    cfg = merge_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data.load()
    fold_of = data.folds.set_index("sample_id").fold
    oof, histories = [], []
    with compute_log() as compute:
        for k in sorted(int(f) for f in fold_of.unique()):
            if folds_to_run is not None and k not in folds_to_run:
                continue
            seed_all(int(cfg["seed"] + k))
            tr, va = fold_of.index[fold_of != k].tolist(), fold_of.index[fold_of == k].tolist()
            model, hist = train_one(data, tr, cfg, device, val_ids=va)
            oof.append(predict_samples(model, data, va, "train", cfg, *model.mean_std, device))
            histories.append(hist.assign(fold=k))
            if verbose:
                print(f"[{name}] fold {k}: val EMD {hist.val_emd.iloc[-1]:.2f}  (train loss {hist.train_loss.iloc[-1]:.2f})")
            if save_checkpoints:
                ckpt = paths.ROOT / "checkpoints" / name
                ckpt.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt / f"fold{k}.pt")
            del model
            torch.cuda.empty_cache()
        test_pred = None
        if predict_test:
            seed_all(int(cfg["seed"] + 100))
            model, _ = train_one(data, fold_of.index.tolist(), cfg, device)
            test_ids = data.photos[data.photos.split == "test"].sample_id.unique()
            test_pred = predict_samples(model, data, test_ids, "test", cfg, *model.mean_std, device)
    oof = pd.concat(oof)
    data_folds = data.folds if folds_to_run is None else data.folds[data.folds.fold.isin(folds_to_run)]
    metrics = save_experiment(name, cfg, oof, data.labels, data_folds, compute, part=part)
    pd.concat(histories).to_csv(paths.EXPERIMENTS / name / "history.csv", index=False)
    if test_pred is not None:
        from .submission import write_submission
        write_submission(test_pred, paths.ROOT / "submissions" / f"{name}.csv")
    if verbose:
        print(f"[{name}] CV {metrics['cv_mean']:.2f} ± {metrics['cv_std']:.2f}  ({compute['hours'] * 60:.1f} min, "
              f"peak {compute.get('peak_mem_gb', 0):.1f} GB)")
    return metrics
