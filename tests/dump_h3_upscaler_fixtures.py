#!/usr/bin/env python3
"""Dump MiniMax-H3 2D latent-upscaler oracles for src/plugins/minimax_h3/upscaler.zig.

Runs the companion node's LatentResizer2D (fp32) on our weights. Not CI.
USER-RUN: clone LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler and pass its
nodes/ path plus the 2D v1 fp32 or fp16 checkpoint.

Usage:
    python3 tests/dump_h3_upscaler_fixtures.py \\
        /path/to/Comfyui_Minimax_h3_latent_Upscaler \\
        /path/to/minimax_h3_latent_upscaler_2d_conv_v1_fp32.pth \\
        [OUT_DIR]

Writes latent_2x.raw / out_2x.raw / latent_1p5.raw / out_1p5.raw (flat f32 BCTHW)
and prints the H3_UPSCALER_FIXTURE export.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

SEED = 0
SHAPE = (1, 24, 3, 4, 6)  # B, C, T, H, W


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    node_root = sys.argv[1]
    ckpt = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "./h3_upscaler_fixtures"
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, os.path.join(node_root, "nodes"))
    import minimax_h3_latent_upscaler_2d as m  # type: ignore

    sd = m._extract_upscaler_sd(m._load_raw_sd(ckpt))
    cfg = m._detect_arch(sd)
    cfg["learned_upsample_ks"] = cfg["learned_upsample_ks"] or None
    model = m.VideoLatentResizer(**{k: v for k, v in cfg.items() if k in m.VideoLatentResizer.__init__.__code__.co_varnames or k in m.LatentResizer.__init__.__code__.co_varnames})
    # Construct from detected cfg the same way load_model does:
    model = m.VideoLatentResizer(
        in_channels=cfg["in_channels"],
        in_blocks=cfg["in_blocks"],
        out_blocks=cfg["out_blocks"],
        channels=cfg["channels"],
        dropout=0.0,
        attn=False,
        temporal_every=cfg["temporal_every"],
        use_multiscale_temporal=cfg["use_multiscale_temporal"],
        temporal_kernel_sizes=cfg["temporal_kernel_sizes"],
        temporal_kernel=cfg["temporal_kernel"],
        use_dual_scale=cfg["use_dual_scale"],
        use_coord=cfg["use_coord"],
        use_se=cfg["use_se"],
        use_large_kernel=cfg["use_large_kernel"],
        large_kernel_size=cfg["large_kernel_size"],
        use_post_refine=cfg["use_post_refine"],
        post_refine_layers=cfg["post_refine_layers"],
        embed_dim=cfg["embed_dim"],
        learned_upsample_ks=cfg["learned_upsample_ks"] or None,
        tier_dim=cfg["tier_dim"],
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"missing keys: {missing[:8]}…")
    model.eval()
    torch.manual_seed(SEED)
    x = torch.randn(*SHAPE, dtype=torch.float32)
    mean = torch.tensor(m.LATENTS_MEAN, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(m.LATENTS_STD, dtype=torch.float32).view(1, -1, 1, 1, 1)
    z = (x - mean) / std
    with torch.no_grad():
        y2 = model(z, scale=2.0, enable_chunking=False, tier=1)
        y15 = model(z, scale=1.5, enable_chunking=False, tier=3)
    y2 = y2 * std + mean
    y15 = y15 * std + mean

    def dump(name: str, t: torch.Tensor) -> str:
        path = os.path.join(out_dir, name)
        t.detach().cpu().contiguous().numpy().astype(np.float32).tofile(path)
        return os.path.abspath(path)

    print("2.0x", tuple(y2.shape), "1.5x", tuple(y15.shape))
    print("export H3_UPSCALER_FIXTURE=" + os.path.abspath(out_dir))
    dump("latent.raw", x)
    dump("out_2x.raw", y2)
    dump("out_1p5.raw", y15)


if __name__ == "__main__":
    main()
