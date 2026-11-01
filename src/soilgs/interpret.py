"""Part 4 / 6: maps (Grad-CAM, attention rollout), shortcut controls and scale robustness for trained models.

Models are re-loaded from `experiments/<name>/config.yaml` + `checkpoints/<name>/fold<k>.pt`, so every
analysis is done on held-out samples of the fold the model never saw.

The scalar explained by the maps is the predicted *coarseness* c = sum_i p_i * log10(d_i), the expected
log grain size under the predicted mass distribution: a positive Grad-CAM region pushes the prediction
towards coarser soil.
"""
import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from . import paths
from .metric import COLS, SIEVES, WEIGHTS
from .models import build_model
from .tiles import TARGET_PPM, tile_grid
from .train import amp_dtype, get_device, resize_to

# Representative diameter of each of the 11 mass bins (geometric mid-point; first bin below 0.002 mm).
BIN_LOG_D = np.log10(np.sqrt(np.concatenate([[0.0005], SIEVES[:-1]]) * SIEVES))


def load_trained(name, fold, device=None):
    """(model, cfg, mean, std) for fold `fold` (or 'full') of experiment `name`."""
    device = device or get_device()
    cfg = yaml.safe_load((paths.EXPERIMENTS / name / "config.yaml").read_text())
    model = build_model(cfg)
    data_cfg = model.pretrained_cfg if hasattr(model, "pretrained_cfg") and cfg.get("pretrained", True) else {}
    mean, std = data_cfg.get("mean", (0.485, 0.456, 0.406)), data_cfg.get("std", (0.229, 0.224, 0.225))
    if cfg.get("lora_r"):
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=2 * cfg["lora_r"], target_modules=["qkv"],
                                                 lora_dropout=0.0, modules_to_save=["head"]))
    ckpt = paths.ROOT / "checkpoints" / name / (f"fold{fold}.pt" if fold != "full" else "full.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval(), cfg, mean, std


def _to_input(tiles, cfg, mean, std, device):
    x = np.stack([resize_to(t.astype(np.float32) / 255, cfg["img_size"]) for t in tiles])
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
    return (x - torch.tensor(mean, device=device).view(1, 3, 1, 1)) / torch.tensor(std, device=device).view(1, 3, 1, 1)


@torch.no_grad()
def photo_probs(model, img, cfg, mean, std, device, tile_fn=None, tiles_subset=None, batch=64):
    """Mean bin probabilities over the grid tiles of one 4.5 px/mm photo (optionally perturbed tiles)."""
    t = int(round(cfg["tile_mm"] * TARGET_PPM))
    boxes = tile_grid(*img.shape[:2], cfg["tile_mm"])
    if tiles_subset is not None:
        boxes = [b for i, b in enumerate(boxes) if tiles_subset(i, b, img.shape)]
    tiles = [img[y:y + t, x:x + t] for y, x, _ in boxes]
    if tile_fn is not None:
        tiles = [tile_fn(tile) for tile in tiles]
    probs = []
    for i in range(0, len(tiles), batch):
        with torch.autocast(device.type, dtype=amp_dtype(), enabled=device.type == "cuda"):
            probs.append(model(_to_input(tiles[i:i + batch], cfg, mean, std, device)).float().softmax(1).cpu().numpy())
    return np.concatenate(probs).mean(0)


def sample_curves(model, images_by_sample, cfg, mean, std, device, **kw):
    """{sample: [img, ...]} -> DataFrame of curves (tile -> photo -> sample averaging, as in training)."""
    out = {s: np.cumsum(np.mean([photo_probs(model, im, cfg, mean, std, device, **kw) for im in imgs], 0)) * 100
           for s, imgs in images_by_sample.items()}
    df = pd.DataFrame(out, index=COLS).T
    df["200"] = 100.0
    return df


def emd_of(curves, labels):
    y = labels.set_index("sample_id").loc[curves.index, COLS].to_numpy(float)
    return pd.Series((np.abs(y - curves[COLS].to_numpy())[:, :-1] * WEIGHTS).sum(1), index=curves.index)


# ------------------------------------------------------------------ perturbations (controls)
def grayscale(tile):
    g = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    return np.repeat(g[..., None], 3, axis=2)


def shuffle_patches(n, seed=0):
    """Cut the tile into n x n patches and permute them: keeps local texture, destroys layout and any
    object larger than one patch (50 mm / n)."""
    rng = np.random.default_rng(seed)

    def fn(tile):
        s = tile.shape[0] // n
        tile = tile[:s * n, :s * n]
        patches = [tile[i * s:(i + 1) * s, j * s:(j + 1) * s] for i in range(n) for j in range(n)]
        order = rng.permutation(len(patches))
        rows = [np.hstack([patches[order[i * n + j]] for j in range(n)]) for i in range(n)]
        return np.vstack(rows)
    return fn


def mean_colour_only(tile):
    """Every pixel replaced by the tile's mean colour: the model sees colour and nothing else."""
    return np.broadcast_to(tile.reshape(-1, 3).mean(0).astype(np.uint8), tile.shape).copy()


def centre_tiles(frac=0.5):
    """Tile filter: keep tiles whose centre lies in the central `frac` of both photo sides."""
    def keep(i, box, shape):
        y, x, t = box
        cy, cx = (y + t / 2) / shape[0], (x + t / 2) / shape[1]
        return abs(cy - .5) < frac / 2 and abs(cx - .5) < frac / 2
    return keep


def border_tiles(frac=0.5):
    inner = centre_tiles(frac)
    return lambda i, b, s: not inner(i, b, s)


# ------------------------------------------------------------------ scale robustness
def at_resolution(img, factor):
    """The same photo as if it had been taken at `factor` x 4.5 px/mm and correctly calibrated: shrink by
    `factor` (detail is lost), then the pipeline resamples back to 4.5 px/mm."""
    if factor >= 1:
        return img
    small = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)


def with_scale_error(img, factor):
    """The same photo with px/mm mis-stated by `factor` (factor > 1: grains appear `factor` x larger)."""
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA)


# ------------------------------------------------------------------ maps
def _coarseness(logits):
    p = logits.float().softmax(1)
    return (p * torch.as_tensor(BIN_LOG_D, device=p.device, dtype=torch.float32)).sum(1)


def gradcam(model, tiles, cfg, mean, std, device):
    """Grad-CAM of predicted coarseness at the last feature map of a timm CNN -> (n, H, W) in [0, 1]."""
    x = _to_input(tiles, cfg, mean, std, device).requires_grad_(False)
    feats = model.forward_features(x)
    feats.retain_grad()
    _coarseness(model.forward_head(feats)).sum().backward()
    w = feats.grad.mean((2, 3), keepdim=True)
    cam = torch.relu((w * feats).sum(1)).detach().cpu().numpy()
    return np.stack([c / (c.max() + 1e-8) for c in cam])


def attention_rollout(model, tiles, cfg, mean, std, device, discard=0.0):
    """Attention rollout (Abnar & Zuidema 2020) for plain ViTs: average heads, add identity, multiply over
    blocks; the [CLS] row over patch tokens is the map -> (n, h, w) in [0, 1]."""
    vit = model.base_model.model if hasattr(model, "base_model") else model
    attns = []
    hooks = []
    for blk in vit.blocks:
        blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: attns.append(o.detach().float().mean(1))))
    with torch.no_grad():
        model(_to_input(tiles, cfg, mean, std, device))
    for h in hooks:
        h.remove()
    n_tok = attns[0].shape[-1]
    roll = torch.eye(n_tok, device=attns[0].device).expand(attns[0].shape[0], -1, -1)
    for a in attns:
        if discard:
            flat = a.flatten(1)
            flat.scatter_(1, flat.topk(int(flat.shape[1] * discard), largest=False).indices, 0)
        a = a + torch.eye(n_tok, device=a.device)
        roll = (a / a.sum(-1, keepdim=True)) @ roll
    pre = getattr(vit, "num_prefix_tokens", 1)
    cls = roll[:, 0, pre:]
    side = int(round(cls.shape[1] ** 0.5))
    maps = cls.reshape(-1, side, side).cpu().numpy()
    return np.stack([(m - m.min()) / (m.max() - m.min() + 1e-8) for m in maps])


def photo_map(model, img, cfg, mean, std, device, method):
    """Stitch per-tile maps into one map over the whole 4.5 px/mm photo."""
    t = int(round(cfg["tile_mm"] * TARGET_PPM))
    boxes = tile_grid(*img.shape[:2], cfg["tile_mm"])
    tiles = [img[y:y + t, x:x + t] for y, x, _ in boxes]
    maps = []
    for i in range(0, len(tiles), 16):
        chunk = tiles[i:i + 16]
        maps.append(gradcam(model, chunk, cfg, mean, std, device) if method == "gradcam"
                    else attention_rollout(model, chunk, cfg, mean, std, device))
    maps = np.concatenate(maps)
    acc, cnt = np.zeros(img.shape[:2], np.float32), np.zeros(img.shape[:2], np.float32)
    for (y, x, _), m in zip(boxes, maps):
        acc[y:y + t, x:x + t] += cv2.resize(m.astype(np.float32), (t, t), interpolation=cv2.INTER_CUBIC)
        cnt[y:y + t, x:x + t] += 1
    return acc / np.maximum(cnt, 1)
