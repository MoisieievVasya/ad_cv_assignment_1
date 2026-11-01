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
    lp_epochs=0, lp_lr=1e-3, lora_r=0, init_weights=None,   # SSL tuning: LP-FT, LoRA, adapted weights
    train_frac=1.0,                                          # data efficiency: share of training samples kept
    epoch_selection=None, inner_folds=3, stop_epoch=None,    # 'nested': pick the stopping epoch by inner CV
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


def get_device():
    """CUDA (local RTX / Colab / Kaggle), Apple-silicon MPS, else CPU. Mixed precision is used on CUDA only."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype():
    """bf16 only where the hardware has it (compute capability >= 8: A100, L4, RTX 30xx+). On a T4 (7.5)
    `is_bf16_supported()` can still return True because of emulation, which is many times slower: use fp16
    (+ GradScaler) there."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def lr_factor(step, total, warmup, schedule):
    if step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def train_one(data, train_ids, cfg, device, val_ids=None):
    """Train on `train_ids`; returns (model, history DataFrame). History has train loss per epoch and,
    if `val_ids` is given, the validation EMD per epoch (monitoring only, never used to pick epochs)."""
    model = build_model(cfg)
    data_cfg = model.pretrained_cfg if hasattr(model, "pretrained_cfg") and cfg.get("pretrained", True) else {}
    mean, std = data_cfg.get("mean", (0.485, 0.456, 0.406)), data_cfg.get("std", (0.229, 0.224, 0.225))
    if cfg["init_weights"]:  # e.g. a domain-adapted (continued DINO) backbone; the head stays new
        state = torch.load(cfg["init_weights"], map_location="cpu")
        missing, _ = model.load_state_dict({k: v for k, v in state.items() if not k.startswith("head")}, strict=False)
        assert all(k.startswith("head") for k in missing), missing
    model = model.to(device)
    if device.type == "cuda" and not any(k in cfg["arch"] for k in ("vit", "deit", "swin", "maxvit")):
        model = model.to(memory_format=torch.channels_last)  # ~20 % faster convolutions under autocast
        torch.backends.cudnn.benchmark = True
    ds = TileTrainSet(data, train_ids, cfg, mean, std)
    if cfg["samples_per_batch"]:
        loader = DataLoader(ds, batch_sampler=GroupedBatchSampler(len(ds.samples), cfg["batch_size"], cfg["steps_per_epoch"],
                                                                  cfg["samples_per_batch"]),
                            num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0)
    else:
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                            num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0)
    history = []
    if cfg["lp_epochs"]:  # LP-FT: first fit the head on the frozen encoder, then fine-tune everything
        lp_cfg = {**cfg, "epochs": cfg["lp_epochs"], "lr": cfg["lp_lr"], "warmup_epochs": 0, "schedule": "constant",
                  "llrd": 1.0, "head_lr_mult": 1.0}
        history += _fit(model, loader, lp_cfg, device, data, val_ids, mean, std, head_only=True, stage="lp")
    history += _fit(model, loader, cfg, device, data, val_ids, mean, std, head_only=cfg["freeze_backbone"],
                    stage="ft", epoch0=len(history))
    model.mean_std = (mean, std)
    return model, pd.DataFrame(history)


def _is_head(name):
    return name.split(".")[0] in ("head", "fc", "classifier")


def _fit(model, loader, cfg, device, data, val_ids, mean, std, head_only=False, stage="ft", epoch0=0):
    for n, p in model.named_parameters():
        if cfg["lora_r"] and not head_only:
            p.requires_grad = "lora_" in n or _is_head(n)
        else:
            p.requires_grad = _is_head(n) or not head_only
    groups = param_groups(model, cfg)
    if cfg["optimizer"] == "sgd":
        opt = torch.optim.SGD(groups, lr=cfg["lr"], momentum=cfg["momentum"], nesterov=True)
    else:
        opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.999))
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
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1
        row = {"stage": stage, "epoch": epoch0 + epoch + 1, "train_loss": float(np.mean(losses))}
        if val_ids is not None:
            pred = predict_samples(model, data, val_ids, "train", {**cfg, "tta": False}, mean, std, device)
            truth = data.labels.set_index("sample_id").loc[pred.index, COLS].to_numpy(float)
            row["val_emd"] = float((np.abs(truth - pred[COLS].to_numpy())[:, :-1] * WEIGHTS).sum(1).mean())
        history.append(row)
        if not math.isfinite(row["train_loss"]):
            break
        if stage == "ft" and cfg["stop_epoch"] and epoch + 1 >= cfg["stop_epoch"]:
            break  # early stop at a pre-chosen epoch; the LR schedule still spans cfg["epochs"]
    return history


def nested_stop_epoch(data, train_ids, cfg, device, site_of):
    """Stopping epoch chosen by an inner group-CV (by site) on the outer-training samples only:
    each inner model logs validation EMD per epoch; the epoch with the lowest mean is returned."""
    from sklearn.model_selection import GroupKFold
    ids = np.array(train_ids)
    curves = []
    for i, (tr, va) in enumerate(GroupKFold(cfg["inner_folds"]).split(ids, groups=site_of.loc[ids].to_numpy())):
        seed_all(int(cfg["seed"] + 50 + i))
        _, hist = train_one(data, ids[tr].tolist(), {**cfg, "stop_epoch": None}, device, val_ids=ids[va].tolist())
        curves.append(hist[hist.stage == "ft"].set_index("epoch").val_emd.reset_index(drop=True))
    mean_curve = pd.concat(curves, axis=1).mean(1)
    return int(mean_curve.idxmin()) + 1, mean_curve.round(2).tolist()


def run_cv(name, cfg, part, folds_to_run=None, predict_test=False, save_checkpoints=False, verbose=True):
    """Full group-CV experiment: OOF curves, per-fold scores, training curves, compute log -> experiments/<name>."""
    cfg = merge_config(cfg)
    device = get_device()
    data = Data.load()
    fold_of = data.folds.set_index("sample_id").fold
    oof, histories, chosen_epochs, inner_curves = [], [], [], []
    with compute_log() as compute:
        for k in sorted(int(f) for f in fold_of.unique()):
            if folds_to_run is not None and k not in folds_to_run:
                continue
            seed_all(int(cfg["seed"] + k))
            tr, va = fold_of.index[fold_of != k].tolist(), fold_of.index[fold_of == k].tolist()
            if cfg["train_frac"] < 1:  # same subsample for every model (seeded by fold only)
                keep = max(2, int(round(cfg["train_frac"] * len(tr))))
                tr = sorted(np.random.default_rng(1000 + k).choice(tr, keep, replace=False).tolist())
            fold_cfg = cfg
            if cfg["epoch_selection"] == "nested":
                stop, inner_curve = nested_stop_epoch(data, tr, cfg, device, data.folds.set_index("sample_id").site)
                chosen_epochs.append(stop)
                inner_curves.append(inner_curve)
                fold_cfg = {**cfg, "stop_epoch": stop}
                seed_all(int(cfg["seed"] + k))
            model, hist = train_one(data, tr, fold_cfg, device, val_ids=va)
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
            full_cfg = {**cfg, "stop_epoch": int(np.median(chosen_epochs))} if chosen_epochs else cfg
            model, _ = train_one(data, fold_of.index.tolist(), full_cfg, device)
            test_ids = data.photos[data.photos.split == "test"].sample_id.unique()
            test_pred = predict_samples(model, data, test_ids, "test", cfg, *model.mean_std, device)
            if save_checkpoints:
                torch.save(model.state_dict(), paths.ROOT / "checkpoints" / name / "full.pt")
    oof = pd.concat(oof)
    data_folds = data.folds if folds_to_run is None else data.folds[data.folds.fold.isin(folds_to_run)]
    extra = {"chosen_epochs": chosen_epochs, "inner_val_curves": inner_curves} if chosen_epochs else None
    metrics = save_experiment(name, cfg, oof, data.labels, data_folds, compute, extra=extra, part=part)
    pd.concat(histories).to_csv(paths.EXPERIMENTS / name / "history.csv", index=False)
    if test_pred is not None:
        from .submission import write_submission
        write_submission(test_pred, paths.ROOT / "submissions" / f"{name}.csv")
    if verbose:
        print(f"[{name}] CV {metrics['cv_mean']:.2f} ± {metrics['cv_std']:.2f}  ({compute['hours'] * 60:.1f} min, "
              f"peak {compute.get('peak_mem_gb', 0):.1f} GB)")
    return metrics


def run_photo_split_cv(name, cfg, part, n_folds=5, seed=0):
    """LEAKY control (HS.1): folds over *photos*, so other photos of a validation sample are in training.
    Held-out photos are predicted per fold; a sample's curve is the mean over all its held-out photos.
    Exists only to show what a non-grouped split does to the score."""
    from dataclasses import replace
    cfg = merge_config(cfg)
    device = get_device()
    data = Data.load()
    train_photos = data.photos[data.photos.split == "train"].reset_index(drop=True)
    fold = np.random.default_rng(seed).permutation(len(train_photos)) % n_folds
    acc, n = {}, {}
    with compute_log() as compute:
        for k in range(n_folds):
            seed_all(int(cfg["seed"] + k))
            tr_ph, va_ph = train_photos[fold != k], train_photos[fold == k]
            d_tr = replace(data, photos=tr_ph)
            model, _ = train_one(d_tr, sorted(tr_ph.sample_id.unique()), cfg, device)
            for pid, s in zip(va_ph.photo_id, va_ph.sample_id):  # one curve per held-out photo
                curve = predict_samples(model, replace(data, photos=va_ph[va_ph.photo_id == pid]), [s], "train",
                                        cfg, *model.mean_std, device)[COLS].to_numpy()[0]
                acc[s], n[s] = acc.get(s, 0) + curve, n.get(s, 0) + 1
            print(f"[{name}] fold {k} done", flush=True)
            del model
            torch.cuda.empty_cache()
    oof = pd.DataFrame({s: acc[s] / n[s] for s in acc}, index=COLS).T
    metrics = save_experiment(name, {**cfg, "split": f"LEAKY random {n_folds}-fold over photos, seed {seed}"},
                              oof, data.labels, data.folds, compute, part=part)
    print(f"[{name}] OOF EMD {metrics['oof_emd']:.2f} (leaky photo split)")
    return metrics
