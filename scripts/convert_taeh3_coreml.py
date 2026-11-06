#!/usr/bin/env python3
"""Install or build MiniMax-H3 taeh3 Core ML preview bundles (MIT weights).

Output layout (matches mlx-serve `h3_tae.zig` + Draw Things install pattern):

    <out>/minimax_h3_tae/
      NOTICE
      weight.bin
      768.mlmodelc … 2048.mlmodelc

Typical flows:

  # Dev: copy a tree you already converted (never commit Draw Things' zip)
  python3 scripts/convert_taeh3_coreml.py --import-tree /path/to/minimax_h3_tae --out ~/.mlx-serve/shared

  # After upstream publishes HF artifacts, fetch_minimax_h3_tae.sh downloads them.
  # Local torch→Core ML export is macOS-only and requires coremltools + taehv checkout:
  python3 scripts/convert_taeh3_coreml.py --export --out ~/.mlx-serve/shared --taehv ~/src/taehv
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / ".cache/taeh3/taeh3.safetensors"
TAEH3_SHA256 = "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13"
VARIANTS = (768, 1024, 1280, 1536, 1792, 2048)

NOTICE = """MiniMax-H3 preview tiny decoder (taeh3) — Core ML bundles for mlx-serve

Weights: taeh3.safetensors from https://huggingface.co/OzzyGT/taeh3
SHA256: 4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13

Copyright (c) 2025 Ollin Boer Bohan — MIT License (see https://github.com/madebyollin/taehv/blob/main/LICENSE)

Converted for mlx-serve H3 step previews (stateful chunk-2 NTHWC graph). Not affiliated
with MiniMax or Draw Things. Do not redistribute third-party compiled assets without
their licenses.
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_root(out: Path) -> Path:
    return out / "minimax_h3_tae"


def validate_layout(root: Path) -> None:
    for v in VARIANTS:
        sub = f"{v}.mlmodelc"
        if not (root / sub).is_dir():
            raise SystemExit(f"missing {root / sub}")
    if not (root / "weight.bin").is_file():
        raise SystemExit(f"missing shared weight.bin under {root}")


def link_weights(root: Path) -> None:
    wb = root / "weight.bin"
    if not wb.is_file():
        raise SystemExit(f"no weight.bin at {wb}")
    for v in VARIANTS:
        weights_dir = root / f"{v}.mlmodelc" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        dst = weights_dir / "weight.bin"
        if dst.is_symlink() or not dst.is_file():
            dst.unlink(missing_ok=True)
        else:
            continue
        try:
            os.link(wb, dst)
        except OSError:
            shutil.copy2(wb, dst)


def write_notice(root: Path) -> None:
    (root / "NOTICE").write_text(NOTICE, encoding="utf-8")


def import_tree(src: Path, out: Path) -> Path:
    dst = bundle_root(out)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    validate_layout(dst)
    link_weights(dst)
    write_notice(dst)
    return dst


def run_fetch_taeh3() -> Path:
    script = ROOT / "scripts/fetch_taeh3.sh"
    subprocess.check_call([str(script)])
    if not DEFAULT_WEIGHTS.is_file():
        raise SystemExit(f"fetch_taeh3 did not produce {DEFAULT_WEIGHTS}")
    got = sha256_file(DEFAULT_WEIGHTS)
    if got != TAEH3_SHA256:
        raise SystemExit(f"taeh3 sha mismatch: {got}")
    return DEFAULT_WEIGHTS


def export_coreml(out: Path, taehv: Path | None, weights: Path | None) -> Path:
    impl = ROOT / "tests/export_taeh3_coreml_impl.py"
    if not impl.is_file():
        raise SystemExit(
            "Core ML export implementation missing. Use --import-tree with a converted "
            "minimax_h3_tae tree, or run fetch_minimax_h3_tae.sh once a HF artifact exists."
        )
    w = weights or run_fetch_taeh3()
    tv = taehv or Path(os.environ.get("TAEHV_DIR", "")).expanduser()
    if not tv.is_dir():
        raise SystemExit("--export needs --taehv PATH or TAEHV_DIR (madebyollin/taehv checkout)")
    dst = bundle_root(out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(impl),
        "--weights",
        str(w),
        "--taehv",
        str(tv),
        "--out",
        str(dst),
    ]
    subprocess.check_call(cmd)
    validate_layout(dst)
    link_weights(dst)
    write_notice(dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="Parent dir (writes minimax_h3_tae/)")
    ap.add_argument("--import-tree", type=Path, help="Copy an existing bundle tree")
    ap.add_argument("--export", action="store_true", help="Run torch→Core ML export (macOS + deps)")
    ap.add_argument("--taehv", type=Path, help="Path to madebyollin/taehv checkout")
    ap.add_argument("--weights", type=Path, help="taeh3.safetensors (default: .cache/taeh3)")
    args = ap.parse_args()

    if args.import_tree:
        root = import_tree(args.import_tree.expanduser().resolve(), args.out.expanduser().resolve())
        print(f"[convert-taeh3] installed {root}")
        return
    if args.export:
        root = export_coreml(args.out.expanduser().resolve(), args.taehv, args.weights)
        print(f"[convert-taeh3] exported {root}")
        return
    ap.error("pass --import-tree or --export")


if __name__ == "__main__":
    main()
