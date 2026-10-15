#!/usr/bin/env python3
"""Download only transformer.safetensors from ddalcu FL2VA 8-bit pack."""

from __future__ import annotations

import os
import sys

from huggingface_hub import hf_hub_download

REPO = "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit"
FILENAME = "transformer.safetensors"
DEFAULT_DIR = os.path.expanduser("~/.mlx-serve/staging/minimax-h3-fl2va-8bit")


def main() -> None:
    dest = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)
    os.makedirs(dest, exist_ok=True)
    out = dest if dest.endswith(".safetensors") else os.path.join(dest, FILENAME)
    if os.path.isfile(out):
        gib = os.path.getsize(out) / (1024**3)
        print(f"already present: {out} ({gib:.2f} GiB)", flush=True)
        return
    print(f"downloading {REPO}/{FILENAME} -> {dest}", flush=True)
    path = hf_hub_download(
        repo_id=REPO,
        filename=FILENAME,
        local_dir=dest,
    )
    gib = os.path.getsize(path) / (1024**3)
    print(f"done: {path} ({gib:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
