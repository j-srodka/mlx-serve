#!/usr/bin/env python3
"""Build a Tokey-style MiniMax-H3 hybrid mlx-serve pack from two 8-bit packs.

FL2VA transformer is the base; later-block ``adaln_proj`` tensors are taken from
REF2VA (default blocks 25-49, matching ``b25-49``). TE, VAEs, tokenizer and
LICENSE are cloned from the REF2VA pack (byte-identical between partitions).

Does not modify either source pack.

Usage:
    uv run --with mlx --with safetensors --with numpy \\
        tests/merge_minimax_h3_hybrid.py \\
        --fl2va-transformer ~/.mlx-serve/staging/minimax-h3-fl2va-8bit/transformer.safetensors \\
        --ref2va-pack ~/.mlx-serve/models/ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit \\
        --out ~/Models/omlx/local/MiniMax-H3-HYBRID-b25-49-MLX-Serve-8bit

In-place into an existing FL2VA pack (no third directory, REF2VA read-only):

    uv run --with mlx --with safetensors --with numpy \\
        tests/merge_minimax_h3_hybrid.py \\
        --in-place-pack ~/Models/omlx/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit \\
        --ref2va-pack ~/Models/omlx/ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit \\
        --ref-blocks 20-49
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import mlx.core as mx

DEFAULT_FL2VA_TRANSFORMER = os.path.expanduser(
    "~/.mlx-serve/staging/minimax-h3-fl2va-8bit/transformer.safetensors"
)
DEFAULT_REF2VA_PACK = os.path.expanduser(
    "~/.mlx-serve/models/ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit"
)
DEFAULT_OUT = os.path.expanduser(
    "~/Models/omlx/local/MiniMax-H3-HYBRID-b25-49-MLX-Serve-8bit"
)

# Byte-identical between FL2VA and REF2VA mlx-serve packs.
CLONE_FROM_REF2VA = [
    "text_encoder.safetensors",
    "video_vae.safetensors",
    "audio_vae.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
    "NOTICE",
    "MODIFICATIONS.md",
    "README.md",
]

# Optional shards present in some local REF2VA packs.
OPTIONAL_CLONE = [
    "image_vae.safetensors",
    "turbo_lora.safetensors",
]

TOKENIZER_OPTIONAL = {
    "LICENSE",
    "NOTICE",
    "MODIFICATIONS.md",
    "README.md",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clone_file(src: Path, dst: Path) -> None:
    """APFS copy-on-write clone, falling back to a real copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        log(f"  exists, skip {dst.name}")
        return
    if subprocess.run(["cp", "-c", str(src), str(dst)]).returncode != 0:
        shutil.copyfile(src, dst)
    log(f"  cloned {dst.name}")


def parse_ref_blocks(spec: str) -> range:
    """Parse ``25-49`` or ``25:50`` into ``range(25, 50)``."""
    spec = spec.strip()
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        start, end = int(lo), int(hi)
    elif ":" in spec:
        start, end = (int(x) for x in spec.split(":", 1))
    else:
        raise SystemExit(f"invalid --ref-blocks {spec!r} (use START-END, e.g. 25-49)")
    if start < 0 or end < start:
        raise SystemExit(f"invalid block range {start}-{end}")
    return range(start, end + 1)


def adaln_prefix(block: int) -> str:
    return f"blocks.{block}.adaln_proj"


def merge_transformer(
    fl2va_path: Path,
    ref2va_transformer: Path,
    out_path: Path,
    ref_blocks: range,
) -> None:
    ref_prefixes = {adaln_prefix(i) for i in ref_blocks}
    log(f"merging transformer: FL2VA base + REF2VA adaln_proj for blocks {ref_blocks.start}-{ref_blocks.stop - 1}")

    # Preload only the REF2VA adaln slice (~25 blocks × 4 tensors).
    ref_tensors: dict[str, mx.array] = {}
    ref_src = mx.load(str(ref2va_transformer))
    for k in list(ref_src.keys()):
        if any(k.startswith(p + ".") for p in ref_prefixes):
            ref_tensors[k] = ref_src.pop(k)
    del ref_src

    missing = [p for p in sorted(ref_prefixes) if not any(k.startswith(p + ".") for k in ref_tensors)]
    if missing:
        raise SystemExit(
            "REF2VA transformer missing adaln_proj for: " + ", ".join(missing)
        )

    out_tensors: dict[str, mx.array] = {}
    n_fl2va = n_ref = n_skipped = 0
    fl2va_src = mx.load(str(fl2va_path))
    for k in list(fl2va_src.keys()):
        arr = fl2va_src.pop(k)
        block_from_ref = any(k.startswith(p + ".") for p in ref_prefixes)
        if block_from_ref:
            out_tensors[k] = ref_tensors[k]
            n_ref += 1
            n_skipped += 1
        else:
            out_tensors[k] = arr
            n_fl2va += 1
        if (n_fl2va + n_ref) % 200 == 0:
            mx.eval(*out_tensors.values())

    del ref_tensors, fl2va_src

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "hybrid": "fl2va_base_ref2va_adaln",
        "ref_blocks": f"{ref_blocks.start}-{ref_blocks.stop - 1}",
        "fl2va_source": str(fl2va_path),
        "ref2va_source": str(ref2va_transformer),
        "merged_by": "mlx-serve tests/merge_minimax_h3_hybrid.py",
    }
    save_path = out_path if str(out_path).endswith(".safetensors") else Path(f"{out_path}.safetensors")
    mx.save_safetensors(str(save_path), out_tensors, metadata=meta)
    gib = save_path.stat().st_size / (1024**3)
    log(
        f"  wrote {out_path.name}: {len(out_tensors)} tensors "
        f"({n_fl2va} FL2VA + {n_ref} REF2VA adaln, skipped {n_skipped} FL2VA adaln) "
        f"-> {gib:.2f} GiB"
    )


def write_hybrid_config(out_dir: Path, ref_blocks: range, ref2va_config: dict) -> None:
    cfg = dict(ref2va_config)
    cfg["partition"] = f"hybrid_b{ref_blocks.start}_{ref_blocks.stop - 1}"
    cfg["tasks"] = ["t2va", "fl2va", "ref2va"]
    cfg["hybrid"] = {
        "base": "fl2va",
        "ref_adaln_from": "ref2va",
        "ref_blocks": f"{ref_blocks.start}-{ref_blocks.stop - 1}",
        "note": "Tokey-style merge: FL2VA transformer except later-block adaln_proj from REF2VA",
    }
    path = out_dir / "config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    log(f"  wrote {path.name} tasks={cfg['tasks']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fl2va-transformer",
        default=DEFAULT_FL2VA_TRANSFORMER,
        help="path to FL2VA 8-bit transformer.safetensors",
    )
    ap.add_argument(
        "--ref2va-pack",
        default=DEFAULT_REF2VA_PACK,
        help="existing REF2VA mlx-serve pack (read-only)",
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="output pack directory")
    ap.add_argument(
        "--ref-blocks",
        default="25-49",
        help="block indices to take adaln_proj from REF2VA (default 25-49)",
    )
    ap.add_argument(
        "--skip-transformer",
        action="store_true",
        help="skip transformer merge if output already exists",
    )
    ap.add_argument(
        "--in-place-pack",
        metavar="DIR",
        help="merge DiT into an existing FL2VA pack: write .hybrid-new, swap, "
        "update config.json only (no sidecar clones)",
    )
    args = ap.parse_args()

    ref_blocks = parse_ref_blocks(args.ref_blocks)
    ref2va_pack = Path(os.path.expanduser(args.ref2va_pack))
    ref2va_transformer = ref2va_pack / "transformer.safetensors"
    ref_config_path = ref2va_pack / "config.json"

    if not ref2va_transformer.is_file():
        raise SystemExit(f"REF2VA transformer not found: {ref2va_transformer}")
    if not ref_config_path.is_file():
        raise SystemExit(f"REF2VA config not found: {ref_config_path}")

    if args.in_place_pack:
        out_dir = Path(os.path.expanduser(args.in_place_pack))
        fl2va_transformer = out_dir / "transformer.safetensors"
        if not out_dir.is_dir():
            raise SystemExit(f"in-place pack not found: {out_dir}")
        if not fl2va_transformer.is_file():
            raise SystemExit(f"FL2VA transformer not found: {fl2va_transformer}")
        log(f"in-place hybrid into {out_dir} (blocks {ref_blocks.start}-{ref_blocks.stop - 1})")
        dit_new = out_dir / "transformer.hybrid-new.safetensors"
        if dit_new.exists():
            raise SystemExit(f"refusing to overwrite stale {dit_new} (remove it first)")
        merge_transformer(fl2va_transformer, ref2va_transformer, dit_new, ref_blocks)
        if not dit_new.is_file():
            raise SystemExit(f"merge did not produce {dit_new}")
        bak = out_dir / "transformer.safetensors.pre-hybrid.bak"
        if bak.exists():
            bak.unlink()
        fl2va_transformer.rename(bak)
        dit_new.rename(fl2va_transformer)
        bak.unlink()
        log("  swapped transformer.safetensors (old DiT removed)")
        with open(ref_config_path) as f:
            ref_config = json.load(f)
        write_hybrid_config(out_dir, ref_blocks, ref_config)
        log("done")
        log(f"  start: MINIMAX_H3_MODEL={out_dir} ./start-minimax-h3-hybrid.sh")
        return

    fl2va_transformer = Path(os.path.expanduser(args.fl2va_transformer))
    out_dir = Path(os.path.expanduser(args.out))
    out_transformer = out_dir / "transformer.safetensors"

    if not fl2va_transformer.is_file():
        raise SystemExit(f"FL2VA transformer not found: {fl2va_transformer}")

    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"output pack: {out_dir}")
    log("cloning shared shards from REF2VA pack (read-only)")
    for name in CLONE_FROM_REF2VA:
        src = ref2va_pack / name
        if not src.is_file():
            if name in TOKENIZER_OPTIONAL:
                log(f"  skip missing optional {name}")
                continue
            raise SystemExit(f"missing in REF2VA pack: {src}")
        clone_file(src, out_dir / name)

    for name in OPTIONAL_CLONE:
        src = ref2va_pack / name
        if src.is_file():
            clone_file(src, out_dir / name)
        else:
            log(f"  skip missing optional {name}")

    if args.skip_transformer and out_transformer.is_file():
        log(f"transformer exists, skip merge ({out_transformer})")
    else:
        merge_transformer(fl2va_transformer, ref2va_transformer, out_transformer, ref_blocks)

    with open(ref_config_path) as f:
        ref_config = json.load(f)
    write_hybrid_config(out_dir, ref_blocks, ref_config)

    log("done")
    log(f"  start: MINIMAX_H3_MODEL={out_dir} ./start-minimax-h3-hybrid.sh")


if __name__ == "__main__":
    main()
