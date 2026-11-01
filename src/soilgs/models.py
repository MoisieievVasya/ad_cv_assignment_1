"""Model factory: timm backbones (CNN / ViT / SSL encoders), a from-scratch ViT, normalisation surgery.

Every model maps a tile (3 x S x S, S = tile_mm * 4.5 px/mm) to 11 logits; softmax gives the mass
fraction per sieve bin, whose cumulative sum is the grain-size curve. Monotonicity, [0, 100] and
100 % at 200 mm therefore hold by construction for every model.
"""
import timm
import torch
import torch.nn as nn

N_BINS = 11


class TinyViT(nn.Module):
    """Plain ViT trained from scratch (Lecture-1 style): conv patch embedding, [CLS], learned positions,
    pre-norm transformer blocks."""

    def __init__(self, img_size=224, patch=16, dim=192, depth=6, heads=3, mlp_ratio=4, drop_path=0.0, n_out=N_BINS):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        n = (img_size // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * .02)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([timm.models.vision_transformer.Block(dim, heads, mlp_ratio, qkv_bias=True, drop_path=d)
                                     for d in dpr])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_out)
        self.num_features = dim

    def forward_features(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x):
        return self.head(self.forward_features(x)[:, 0])


def bn_to_gn(module, groups=32):
    """Replace every BatchNorm2d by GroupNorm (affine params copied); batch-independent statistics."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            g = groups if child.num_features % groups == 0 else 1
            gn = nn.GroupNorm(g, child.num_features, affine=True)
            with torch.no_grad():
                gn.weight.copy_(child.weight)
                gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            bn_to_gn(child, groups)
    return module


def freeze_bn_stats(module):
    """Keep BatchNorm running statistics (from ImageNet) fixed: call after every model.train()."""
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def build_model(cfg):
    """cfg keys: arch, pretrained, norm ('bn' | 'frozen_bn' | 'gn'), drop_path, img_size, vit_* for TinyViT."""
    if cfg["arch"] == "tiny_vit":
        return TinyViT(img_size=cfg["img_size"], patch=cfg.get("patch", 16), dim=cfg.get("vit_dim", 192),
                       depth=cfg.get("vit_depth", 6), heads=cfg.get("vit_heads", 3), drop_path=cfg.get("drop_path", 0.0))
    kwargs = {"pretrained": cfg.get("pretrained", True), "num_classes": N_BINS}
    if cfg.get("drop_path"):
        kwargs["drop_path_rate"] = cfg["drop_path"]
    if any(k in cfg["arch"] for k in ("vit", "deit", "swin", "maxvit", "dinov2", "eva")):
        kwargs["img_size"] = cfg["img_size"]
    model = timm.create_model(cfg["arch"], **kwargs)
    if cfg.get("norm") == "gn":
        bn_to_gn(model)
    return model


def param_groups(model, cfg):
    """AdamW/SGD parameter groups: no weight decay on norms, biases, positional/cls tokens; optional
    layer-wise LR decay (LLRD) for transformers (block i gets lr * decay^(depth - i))."""
    wd, lr, decay = cfg["weight_decay"], cfg["lr"], cfg.get("llrd", 1.0)
    blocks = [n for n, _ in model.named_parameters() if ".blocks." in n or n.startswith("blocks.")]
    depth = 1 + max((int(n.split("blocks.")[1].split(".")[0]) for n in blocks), default=-1)
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Norm weights, biases (ndim <= 1) and position / class tokens get no weight decay unless the
        # ablation `wd_on_all` asks for it.
        no_wd = not cfg.get("wd_on_all", False) and (
            p.ndim <= 1 or any(k in n for k in ("pos", "cls", "token", "relative_position")))
        if decay < 1 and depth:
            if "blocks." in n:
                layer = int(n.split("blocks.")[1].split(".")[0]) + 1
            elif any(k in n for k in ("patch_embed", "pos", "cls", "token")):
                layer = 0
            else:
                layer = depth + 1
            scale = decay ** (depth + 1 - layer)
        else:
            scale = 1.0
        if n.split(".")[0] in ("head", "fc", "classifier"):
            scale *= cfg.get("head_lr_mult", 1.0)
        key = (scale, 0.0 if no_wd else wd)
        groups.setdefault(key, []).append(p)
    return [{"params": ps, "lr": lr * s, "lr_scale": s, "weight_decay": w} for (s, w), ps in groups.items()]
