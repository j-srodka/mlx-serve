#!/usr/bin/env python3
"""Torch→Core ML export for taeh3 (macOS). Invoked by scripts/convert_taeh3_coreml.py --export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VARIANTS = (768, 1024, 1280, 1536, 1792, 2048)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--taehv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    try:
        import coremltools as ct  # noqa: F401
        import torch
    except ImportError as e:
        raise SystemExit(
            "export needs macOS with: pip install coremltools torch safetensors"
        ) from e

    sys.path.insert(0, str(args.taehv.resolve()))
    try:
        from taehv import TAEHV  # type: ignore
    except ImportError as e:
        raise SystemExit(f"could not import taehv from {args.taehv}") from e

    raise SystemExit(
        "Stateful chunk-2 Core ML export is not checked in yet. "
        "Convert locally (coremltools 9.x, match Draw Things I/O: latent NTHWC T=2, "
        "act_0…act_8), then install with:\n"
        f"  python3 scripts/convert_taeh3_coreml.py --import-tree <your/minimax_h3_tae> --out {args.out.parent}"
    )


if __name__ == "__main__":
    main()
