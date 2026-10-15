#!/usr/bin/env python3
"""Download Mamad8 Image VAE next to the Ref2VA pack. Never overwrite video_vae."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

WANT_SHA = "6c3d0bfa055986a803a566a862fcde283a1e63db62829e5ef4a2a5aebf50bb86"
PACK = Path.home() / ".mlx-serve/models/ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit"
DEST = PACK / "image_vae.safetensors"
STAGE = Path.home() / ".mlx-serve/models/Mamad8/MiniMax-H3-Image-VAE"
SRC_NAME = "minimax_h3_t1_image_vae_step1597.safetensors"
VIDEO = PACK / "video_vae.safetensors"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not VIDEO.is_file():
        print("missing video_vae — refusing to write into pack", file=sys.stderr)
        return 1
    if DEST.is_file() and DEST.stat().st_size > 0:
        got = sha256(DEST)
        print(f"already present {DEST} ({DEST.stat().st_size} bytes)")
        print(f"sha256 {got}")
        if got != WANT_SHA:
            print(f"WARNING sha mismatch want {WANT_SHA}", file=sys.stderr)
            return 1
        return 0

    STAGE.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf",
        "download",
        "Mamad8/MiniMax-H3-Image-VAE",
        SRC_NAME,
        "--local-dir",
        str(STAGE),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    src = STAGE / SRC_NAME
    if not src.is_file():
        print(f"download did not produce {src}", file=sys.stderr)
        return 1
    got = sha256(src)
    print(f"downloaded sha256 {got}")
    if got != WANT_SHA:
        print(f"sha mismatch want {WANT_SHA}", file=sys.stderr)
        return 1
    try:
        os.link(src, DEST)
        print(f"hardlinked {src} -> {DEST}")
    except OSError:
        shutil.copyfile(src, DEST)
        print(f"copied {src} -> {DEST}")
    print(f"image_vae {DEST.stat().st_size} video_vae {VIDEO.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
