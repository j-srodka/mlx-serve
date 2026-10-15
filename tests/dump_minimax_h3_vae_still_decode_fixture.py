#!/usr/bin/env python3
"""Dump T=1 still-decode constants + a synthetic latent for Image VAE parity.

The encoder fixture already dumps T=1 latents (`x` / `latent`). This file
records the decode contract ComfyUI T=1 stills rely on: one ViT3D pass, no
temporal chunk pad, keep the single frame after FRAME_PRE_PADDING.

Usage (optional, needs torch + the Image VAE):
  python3 tests/dump_minimax_h3_vae_still_decode_fixture.py \
    --vae ~/.mlx-serve/models/ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit/image_vae.safetensors \
    --out /tmp/minimax_h3_still_decode_fixture.safetensors

Without --vae, writes JSON constants only (--constants-out).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Match src/minimax_h3_vae.zig
VAE_RATIO_T = 4
FRAME_PRE_PADDING = 3
TOKENS_CHUNK_SIZE = 5
TOKEN_OVERLAP = 2

CONSTANTS = {
    "VAE_RATIO_T": VAE_RATIO_T,
    "FRAME_PRE_PADDING": FRAME_PRE_PADDING,
    "TOKENS_CHUNK_SIZE": TOKENS_CHUNK_SIZE,
    "TOKEN_OVERLAP": TOKEN_OVERLAP,
    "pixel_frames_after_pre_padding": VAE_RATIO_T - FRAME_PRE_PADDING,
    "latent_t": 1,
    "note": "decodeStill must NOT call planTemporal; T=1 padded to a 5-token chunk is the grid artifact.",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vae", help="Mamad8 image_vae.safetensors (optional encode dump)")
    ap.add_argument("--out", help="safetensors dump path (requires --vae + torch)")
    ap.add_argument(
        "--constants-out",
        default=str(Path(__file__).resolve().parent.parent / "src/fixtures/minimax_h3_still_decode.json"),
        help="JSON constants path",
    )
    args = ap.parse_args()

    const_path = Path(args.constants_out)
    const_path.parent.mkdir(parents=True, exist_ok=True)
    const_path.write_text(json.dumps(CONSTANTS, indent=2) + "\n")
    print("wrote", const_path)

    if CONSTANTS["pixel_frames_after_pre_padding"] != 1:
        print("constants broken: T=1 would not yield one pixel frame", file=sys.stderr)
        return 1

    if not args.vae:
        return 0

    # Encode-only dump: reuse the encoder fixture's T=1 path so a later zig
    # test can feed the same latent into decodeStill. Full pixel dump vs
    # ComfyUI needs ComfyUI (out of scope for this backend-only stills path).
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from dump_minimax_h3_vae_encoder_fixture import (  # noqa: E402
        EncoderFCN3D,
        IMAGENET_MEAN,
        IMAGENET_STD,
        LATENTS_MEAN_KEY,
        LATENTS_STD_KEY,
        synth_image,
        tiled_encode,
    )
    import torch
    import torch.nn as nn
    from safetensors import safe_open
    from safetensors.torch import save_file

    enc = EncoderFCN3D(
        ch=128,
        ch_mult=[1, 2, 2, 4, 4, 8],
        space_down=[2, 2, 2, 2, 1, 1],
        time_down=[1, 2, 2, 1, 1, 1],
        num_res_blocks=2,
        in_channels=3,
        z_channels=24,
        double_z=True,
    )
    quant = nn.Conv3d(48, 48, 1)
    tensors = {}
    with safe_open(args.vae, framework="pt") as f:
        for k in f.keys():
            if k.startswith("encoder.") or k.startswith("quant_conv.") or k in (
                LATENTS_MEAN_KEY,
                LATENTS_STD_KEY,
            ):
                tensors[k] = f.get_tensor(k).float()
    enc.load_state_dict({k[len("encoder.") :]: v for k, v in tensors.items() if k.startswith("encoder.")})
    quant.load_state_dict(
        {k[len("quant_conv.") :]: v for k, v in tensors.items() if k.startswith("quant_conv.")}
    )
    enc.eval()
    quant.eval()
    lm = tensors[LATENTS_MEAN_KEY].view(1, -1, 1, 1, 1)
    ls = tensors[LATENTS_STD_KEY].view(1, -1, 1, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)

    def encode(x_pm1):
        x = (x_pm1 + 1.0) * 0.5
        x = (x - mean) / std
        with torch.no_grad():
            moments = tiled_encode(lambda z: quant(enc(z)), x)
        m = torch.chunk(moments, 2, dim=1)[0]
        return (m - lm) / ls

    img = synth_image(128, seed=3)
    lat = encode(img)
    if tuple(lat.shape[2:3]) != (1,):
        print(f"expected latent T=1, got {tuple(lat.shape)}", file=sys.stderr)
        return 1
    out = {"x": img.contiguous(), "latent": lat.float().contiguous()}
    if not args.out:
        print("pass --out to write safetensors", file=sys.stderr)
        return 1
    save_file(out, args.out)
    print(f"wrote {args.out} image {tuple(img.shape)} latent {tuple(lat.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
