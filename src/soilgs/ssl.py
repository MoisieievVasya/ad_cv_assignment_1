"""Self-supervised encoders: frozen features (k-NN / ridge / linear probe), CKA, continued DINO pre-training.

Encoders see the same 50 mm tiles as every other model, resized 225 -> 224 px (patch 14 and 16 both
divide 224; the 0.4 % resize is negligible next to the 2x scale errors studied in Block 0).
"""
import copy
import math

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import paths
from .metric import COLS, WEIGHTS, emd
from .targets import decode, encode
from .tiles import TARGET_PPM, tile_grid
from .train import Data, amp_dtype, resize_to

ENCODERS = {
    "dinov2_s": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov3_s": "vit_small_patch16_dinov3.lvd1689m",
    "mae_b": "vit_base_patch16_224.mae",
}
INPUT_PX = 224


def load_encoder(key, weights=None, dynamic=False):
    model = timm.create_model(ENCODERS[key], pretrained=True, num_classes=0, img_size=INPUT_PX, dynamic_img_size=dynamic)
    if weights:
        state = torch.load(weights, map_location="cpu")
        model.load_state_dict({k: v for k, v in state.items() if not k.startswith("head")}, strict=False)
    cfg = model.pretrained_cfg
    return model, cfg["mean"], cfg["std"]


def tile_batches(data, tile_mm=50, batch_size=64):
    """Yields (meta rows, uint8 tiles) for every grid tile of every photo, train and test."""
    t = int(round(tile_mm * TARGET_PPM))
    for r in data.photos.itertuples():
        img = data.images[r.photo_id]
        boxes = tile_grid(*img.shape[:2], tile_mm)
        for i in range(0, len(boxes), batch_size):
            chunk = boxes[i:i + batch_size]
            yield ([dict(photo_id=r.photo_id, sample_id=r.sample_id, split=r.split, tile=i + j) for j in range(len(chunk))],
                   np.stack([img[y:y + t, x:x + t] for y, x, _ in chunk]))


@torch.no_grad()
def encode_tiles(model, tiles_u8, mean, std, device):
    """CLS token and mean patch token, concatenated (2 x dim)."""
    x = np.stack([resize_to(t.astype(np.float32) / 255, INPUT_PX) for t in tiles_u8])
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
    x = (x - torch.tensor(mean, device=device).view(1, 3, 1, 1)) / torch.tensor(std, device=device).view(1, 3, 1, 1)
    with torch.autocast(device.type, dtype=amp_dtype(), enabled=device.type == "cuda"):
        tokens = model.forward_features(x).float()
    n_prefix = getattr(model, "num_prefix_tokens", 1)
    return torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(1)], 1).cpu().numpy()


def tile_features(key, data=None, weights=None, tag=None, tile_mm=50):
    """Cached per-tile features: (meta DataFrame, float32 array)."""
    tag = tag or key
    path = paths.CACHE / f"feats_{tag}_{tile_mm}mm.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return pd.DataFrame(z["meta"].tolist()), z["X"]
    data = data or Data.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = load_encoder(key, weights)
    model = model.to(device).eval()
    meta, feats = [], []
    for rows, tiles in tile_batches(data, tile_mm):
        meta += rows
        feats.append(encode_tiles(model, tiles, mean, std, device))
    X = np.concatenate(feats).astype(np.float32)
    np.savez(path, meta=np.array(meta, dtype=object), X=X)
    return pd.DataFrame(meta), X


def sample_matrix(meta, X):
    """Tile -> photo mean -> sample mean and std of tile features (sample-level design matrix)."""
    df = pd.DataFrame(X, index=pd.MultiIndex.from_frame(meta[["split", "sample_id", "photo_id"]]))
    photo_mean = df.groupby(level=[0, 1, 2]).mean()
    mean = photo_mean.groupby(level=[0, 1]).mean()
    std = df.groupby(level=[0, 1]).std().fillna(0)
    return pd.concat([mean, std], axis=1, keys=["mean", "std"])


def _ridge_fit_predict(Xtr, Ytr, Xte, alpha, mode):
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha, solver="lsqr").fit(sc.transform(Xtr), encode(Ytr, mode))
    return decode(m.predict(sc.transform(Xte)), mode)


def _knn_predict(Xtr, Ytr, Xte, k):
    sc = StandardScaler().fit(Xtr)
    a, b = sc.transform(Xtr), sc.transform(Xte)
    a, b = a / np.linalg.norm(a, axis=1, keepdims=True), b / np.linalg.norm(b, axis=1, keepdims=True)
    idx = np.argsort(-(b @ a.T), axis=1)[:, :k]
    return Ytr[idx].mean(1)


def frozen_oof(S, labels, folds, method="ridge", mode="sqrt_mass", alphas=(1, 10, 100, 1000, 10000), ks=(1, 3, 5)):
    """OOF sample curves; the hyper-parameter (alpha or k) is chosen by an inner leave-one-site-out CV on
    the training samples of each outer fold (nested — the outer fold never influences the choice)."""
    S = S.loc["train"]
    lab = labels.set_index("sample_id")
    fm = folds.set_index("sample_id")
    ids = S.index.tolist()
    X, Y = S.to_numpy(float), lab.loc[ids, COLS].to_numpy(float)
    fold, site = fm.loc[ids, "fold"].to_numpy(), fm.loc[ids, "site"].to_numpy()
    grid = alphas if method == "ridge" else ks
    predict = (lambda a, b, c, h: _ridge_fit_predict(a, b, c, h, mode)) if method == "ridge" else _knn_predict
    oof, chosen = np.zeros_like(Y), []
    for k in np.unique(fold):
        tr, va = fold != k, fold == k
        inner = []
        for h in grid:
            P = np.zeros((tr.sum(), Y.shape[1]))
            for s in np.unique(site[tr]):
                i_va = site[tr] == s
                P[i_va] = predict(X[tr][~i_va], Y[tr][~i_va], X[tr][i_va], h)
            inner.append(emd(Y[tr], P))
        best = grid[int(np.argmin(inner))]
        chosen.append(best)
        oof[va] = predict(X[tr], Y[tr], X[va], best)
    return pd.DataFrame(oof, index=ids, columns=COLS), chosen


def frozen_test(S, labels, method, h, mode="sqrt_mass"):
    lab = labels.set_index("sample_id")
    tr, te = S.loc["train"], S.loc["test"]
    Y = lab.loc[tr.index, COLS].to_numpy(float)
    P = _ridge_fit_predict(tr.to_numpy(float), Y, te.to_numpy(float), h, mode) if method == "ridge" else \
        _knn_predict(tr.to_numpy(float), Y, te.to_numpy(float), h)
    return pd.DataFrame(P, index=te.index, columns=COLS)


def linear_probe_oof(meta, X, labels, folds, epochs=200, lr=1e-2, wd=1e-3, seed=0):
    """Tile-level linear softmax head on frozen features, EMD loss (same head/loss as fine-tuning)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lab = labels.set_index("sample_id")
    tr_mask = (meta.split == "train").to_numpy()
    m = meta[tr_mask].reset_index(drop=True)
    Xt = X[tr_mask]
    curves = lab.loc[m.sample_id, COLS].to_numpy(float)
    mass = np.diff(np.column_stack([np.zeros(len(curves)), curves]), axis=1).clip(0) / 100
    fold = m.sample_id.map(folds.set_index("sample_id").fold).to_numpy()
    w = torch.as_tensor(WEIGHTS, dtype=torch.float32, device=device)
    out = []
    for k in np.unique(fold):
        torch.manual_seed(seed + int(k))
        tr, va = fold != k, fold == k
        mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-6
        xtr = torch.tensor((Xt[tr] - mu) / sd, device=device)
        ytr = torch.tensor(mass[tr], dtype=torch.float32, device=device)
        head = nn.Linear(X.shape[1], 11).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            for idx in torch.randperm(len(xtr), device=device).split(256):
                p = head(xtr[idx]).softmax(1)
                loss = ((p.cumsum(1) - ytr[idx].cumsum(1)).abs()[:, :-1] * 100 * w).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            p = head(torch.tensor((Xt[va] - mu) / sd, device=device)).softmax(1).cpu().numpy()
        df = m[va][["photo_id", "sample_id"]].copy()
        df[[f"p{i}" for i in range(11)]] = p
        s = df.groupby(["photo_id", "sample_id"]).mean(numeric_only=True).groupby("sample_id").mean()
        out.append(pd.DataFrame(np.cumsum(s.to_numpy(), 1) * 100, index=s.index, columns=COLS))
    return pd.concat(out)


# ---------------------------------------------------------------- CKA
def linear_cka(A, B):
    """Linear CKA (Kornblith et al. 2019) between two (n, d) representations of the same inputs."""
    A, B = A - A.mean(0), B - B.mean(0)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


@torch.no_grad()
def block_outputs(model, tiles_u8, mean, std, device):
    """Mean-token output of every transformer block -> list of (n, d) arrays."""
    feats = []
    hooks = [blk.register_forward_hook(lambda m, i, o: feats.append(o.float().mean(1).cpu().numpy())) for blk in model.blocks]
    encode_tiles(model, tiles_u8, mean, std, device)
    for h in hooks:
        h.remove()
    return feats


def cka_by_block(model_a, model_b, tiles_u8, mean, std, device, batch=64):
    outs = []
    for model in (model_a, model_b):
        model.to(device).eval()
        per = [block_outputs(model, tiles_u8[i:i + batch], mean, std, device) for i in range(0, len(tiles_u8), batch)]
        outs.append([np.concatenate([p[j] for p in per]) for j in range(len(per[0]))])
        model.cpu()
    return [linear_cka(a, b) for a, b in zip(*outs)]


# ---------------------------------------------------------------- continued DINO pre-training
class DINOHead(nn.Module):
    def __init__(self, dim, out_dim=4096, hidden=1024, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, out_dim, bias=False))

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


def dino_views(img, rng, mode, n_local=4, global_px=224, local_px=98):
    """Two global + n local views of one photo.

    mode='fixed_scale': every view is a crop at the cached 4.5 px/mm, never resized (global = 50 mm,
      local = 22 mm, locals taken inside the globals' area). Scale is preserved — label-safe.
    mode='rrc': standard DINO multi-crop, RandomResizedCrop from a 100 mm region (global area 32-100 %,
      local 5-32 %), resized to 224 / 98 px — mm/px changes by up to ~4x between views.
    """
    H, W = img.shape[:2]
    def flip(v):
        v = v[:, ::-1] if rng.random() < .5 else v
        return np.rot90(v, rng.integers(4))
    if mode == "fixed_scale":
        span = min(int(global_px * 1.5), H, W)
        y0, x0 = rng.integers(H - span + 1), rng.integers(W - span + 1)
        g = [img[y0 + a:y0 + a + global_px, x0 + b:x0 + b + global_px]
             for a, b in rng.integers(0, span - global_px + 1, size=(2, 2))]
        l = [img[y0 + a:y0 + a + local_px, x0 + b:x0 + b + local_px]
             for a, b in rng.integers(0, span - local_px + 1, size=(n_local, 2))]
    else:
        region = min(int(100 * TARGET_PPM), H, W)
        y0, x0 = rng.integers(H - region + 1), rng.integers(W - region + 1)
        R = img[y0:y0 + region, x0:x0 + region]
        def rrc(lo, hi, out):
            s = int(region * math.sqrt(rng.uniform(lo, hi)))
            a, b = rng.integers(region - s + 1), rng.integers(region - s + 1)
            return resize_to(R[a:a + s, b:b + s].astype(np.float32), out)
        g = [rrc(.32, 1, global_px) for _ in range(2)]
        l = [rrc(.05, .32, local_px) for _ in range(n_local)]
    return [np.ascontiguousarray(flip(v)) for v in g], [np.ascontiguousarray(flip(v)) for v in l]


def dino_adapt(key, mode, epochs=10, steps_per_epoch=100, batch_size=32, lr=2e-5, head_lr=1e-3, head_warmup_epochs=1,
               teacher_momentum=0.996, out_name=None, monitor=None, seed=0):
    """Continue DINO self-distillation of a pretrained encoder on unlabelled *training* tiles only.
    Returns a history DataFrame (DINO loss per epoch + optional monitor metric) and the weights path."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda")
    data = Data.load(splits=("train",))
    imgs = list(data.images.values())
    student, mean, std = load_encoder(key, dynamic=True)  # local views are 98 px
    teacher = copy.deepcopy(student)
    dim = student.num_features
    s_head = DINOHead(dim)
    t_head = copy.deepcopy(s_head)
    student, teacher, s_head, t_head = (m.to(device) for m in (student, teacher, s_head, t_head))
    for p in list(teacher.parameters()) + list(t_head.parameters()):
        p.requires_grad = False
    opt = torch.optim.AdamW([{"params": student.parameters(), "lr": lr, "weight_decay": 0.04},
                             {"params": s_head.parameters(), "lr": head_lr, "weight_decay": 0.04}])
    center = torch.zeros(1, 4096, device=device)
    mean_t, std_t = torch.tensor(mean, device=device).view(1, 3, 1, 1), torch.tensor(std, device=device).view(1, 3, 1, 1)
    to_t = lambda vs: (torch.from_numpy(np.stack(vs).astype(np.float32) / 255).permute(0, 3, 1, 2).to(device) - mean_t) / std_t
    history, total = [], epochs * steps_per_epoch
    step = 0
    for epoch in range(epochs):
        backbone_on = epoch >= head_warmup_epochs  # random head first, so it cannot wreck the encoder
        losses = []
        for _ in range(steps_per_epoch):
            G, L = [[], []], [[] for _ in range(4)]
            for _ in range(batch_size):
                g, l = dino_views(imgs[rng.integers(len(imgs))], rng, mode)
                for i in range(2):
                    G[i].append(g[i])
                for i in range(4):
                    L[i].append(l[i])
            g_t, l_t = [to_t(v) for v in G], [to_t(v) for v in L]
            with torch.autocast("cuda", dtype=amp_dtype()):
                with torch.no_grad():
                    t_out = [t_head(teacher(v)).float() for v in g_t]
                s_out = [s_head(student(v)).float() for v in g_t + l_t]
            t_prob = [F.softmax((t - center) / 0.04, -1) for t in t_out]
            loss, n = 0, 0
            for ti, tp in enumerate(t_prob):
                for si, so in enumerate(s_out):
                    if si == ti:
                        continue
                    loss = loss + torch.sum(-tp * F.log_softmax(so / 0.1, -1), -1).mean()
                    n += 1
            loss = loss / n
            for g_ in opt.param_groups[:1]:
                g_["lr"] = lr * (0.5 * (1 + math.cos(math.pi * step / total))) if backbone_on else 0.0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(s_head.parameters()), 3.0)
            opt.step()
            with torch.no_grad():
                m = 1 - (1 - teacher_momentum) * (math.cos(math.pi * step / total) + 1) / 2
                for ps, pt in zip(list(student.parameters()) + list(s_head.parameters()),
                                  list(teacher.parameters()) + list(t_head.parameters())):
                    pt.mul_(m).add_(ps.detach(), alpha=1 - m)
                center = center * 0.9 + torch.cat(t_out).mean(0, keepdim=True) * 0.1
            losses.append(loss.item())
            step += 1
        row = {"epoch": epoch + 1, "dino_loss": float(np.mean(losses)), "backbone_trained": backbone_on}
        if monitor is not None:
            row.update(monitor(teacher, mean, std))
        history.append(row)
        print(row, flush=True)
    out = paths.ROOT / "checkpoints" / f"{out_name or f'dino_adapt_{key}_{mode}'}.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save(teacher.state_dict(), out)  # the teacher (EMA) is the encoder DINO keeps
    return pd.DataFrame(history), out
