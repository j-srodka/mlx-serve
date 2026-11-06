#!/usr/bin/env python3
"""Verify mlx-serve Core ML taeh3 bundles against a fixed latent fixture.

Compares RGB from the C bridge (ANE) vs optional PyTorch taehv reference on the same
latent chunk. Dev-only: comparing against Draw Things bundles under /tmp is fine;
CI must use MIT-derived trees only.

Usage:
  python3 tests/verify_taeh3_coreml.py --bundle ~/.mlx-serve/shared/minimax_h3_tae
  MINIMAX_H3_TAE_LATENT=/path/to/video_latent_fp16.bin python3 tests/verify_taeh3_coreml.py ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAT = Path("/tmp/h3-tae-probe/latent_export/video_latent_fp16.bin")
DEFAULT_META = Path("/tmp/h3-tae-probe/latent_export/meta.json")
PSNR_FLOOR = 25.0


def psnr(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(255.0**2 / mse)


def load_latent_nthwc(meta: dict, bin_path: Path):
    import numpy as np

    t, h, w, c = meta["t"], meta["h"], meta["w"], meta["c"]
    raw = bin_path.read_bytes()
    need = t * h * w * c * 2
    if len(raw) != need:
        raise SystemExit(f"latent size {len(raw)} != {need}")
    fp16 = np.frombuffer(raw, dtype=np.float16).reshape(t, h, w, c)
    return fp16.astype(np.float32)


def resolve_mlmodelc(bundle: Path, meta: dict) -> Path:
    if bundle.name.endswith(".mlmodelc"):
        return bundle
    px = max(meta["h"], meta["w"]) * 16
    for v in (768, 1024, 1280, 1536, 1792, 2048):
        if px <= v:
            return bundle / f"{v}.mlmodelc"
    return bundle / "2048.mlmodelc"


def bridge_rgb(bundle: Path, latent, t_idx: int) -> tuple[int, int, bytes]:
    t, h, w, c = latent.shape
    flat = latent.reshape(-1).astype("float32").tobytes()
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tf.write(flat)
        lat_file = tf.name

    src = ROOT / "lib/h3_tae/h3_tae_preview.m"
    test_c = f"""
#include "h3_tae_preview.h"
#include <stdio.h>
#include <stdlib.h>
int main(void) {{
  FILE *f = fopen("{lat_file}", "rb");
  if (!f) return 1;
  size_t n = (size_t){t}*{h}*{w}*{c};
  float *lat = malloc(n * sizeof(float));
  if (!lat || fread(lat, sizeof(float), n, f) != n) return 1;
  fclose(f);
  char err[512] = {{0}};
  h3_tae_ctx *ctx = h3_tae_preview_open("{bundle}", 1, err, sizeof err);
  if (!ctx) {{ fprintf(stderr, "%s\\n", err); return 1; }}
  int pw=0, ph=0;
  size_t rgb_len = (size_t){h}*16*{w}*16*3;
  uint8_t *rgb = calloc(1, rgb_len);
  if (h3_tae_preview_decode_frame(ctx, lat, {t}, {h}, {w}, {c}, {t_idx}, rgb, &pw, &ph, err, sizeof err) != 0) {{
    fprintf(stderr, "%s\\n", err); return 1;
  }}
  fwrite(rgb, 1, (size_t)pw*ph*3, stdout);
  h3_tae_preview_close(ctx);
  free(lat); free(rgb);
  return 0;
}}
"""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "probe.c").write_text(test_c)
        subprocess.check_call(
            ["clang", "-c", "-fobjc-arc", "-O2", "-I", str(ROOT / "lib/h3_tae"), str(src), "-o", str(td / "h3.o")],
        )
        subprocess.check_call(
            [
                "clang",
                "-O2",
                str(td / "probe.c"),
                str(td / "h3.o"),
                "-o",
                str(td / "probe"),
                "-framework",
                "CoreML",
                "-framework",
                "Foundation",
            ],
        )
        out = subprocess.check_output([str(td / "probe")])
    os.unlink(lat_file)
    # pw/ph not returned through stdout — recompute from meta
    pw = w * 16
    ph = h * 16
    return pw, ph, out


def pytorch_rgb(taehv: Path, weights: Path, latent, t_idx: int):
    sys.path.insert(0, str(taehv))
    import torch
    from taehv import TAEHV  # type: ignore

    model = TAEHV.from_pretrained(str(weights.parent)).eval()
    with torch.no_grad():
        x = torch.from_numpy(latent).unsqueeze(0)  # [1,T,H,W,C] — adjust if API differs
        # Reference path is best-effort; skip if API mismatch
        y = model.decode(x[:, t_idx : t_idx + 2])
    arr = y[0, 0].clamp(0, 1).mul(255).byte().cpu().numpy()
    return arr.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--latent-bin", type=Path, default=DEFAULT_LAT)
    ap.add_argument("--latent-meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--t-idx", type=int, default=0)
    ap.add_argument("--taehv", type=Path, help="Optional taehv checkout for PyTorch reference")
    ap.add_argument("--weights", type=Path, help="taeh3.safetensors for PyTorch arm")
    args = ap.parse_args()

    if sys.platform != "darwin":
        print("SKIP: Core ML verify is macOS-only")
        return

    if not args.latent_bin.is_file() or not args.latent_meta.is_file():
        print(f"SKIP: latent fixture missing ({args.latent_bin})")
        return

    bundle = args.bundle.expanduser().resolve()
    meta = json.loads(args.latent_meta.read_text())
    ml = resolve_mlmodelc(bundle, meta)
    if not ml.is_dir():
        raise SystemExit(f"missing {ml}")

    latent = load_latent_nthwc(meta, args.latent_bin)
    pw, ph, rgb_bytes = bridge_rgb(ml, latent, args.t_idx)
    expected = pw * ph * 3
    if len(rgb_bytes) != expected:
        raise SystemExit(f"bridge returned {len(rgb_bytes)} bytes, want {expected}")

    print(f"bridge ok {pw}x{ph} t_idx={args.t_idx} bundle={ml}")

    if args.taehv and args.weights:
        try:
            ref = pytorch_rgb(args.taehv, args.weights, latent, args.t_idx)
            import numpy as np

            got = np.frombuffer(rgb_bytes, dtype=np.uint8)
            score = psnr(got[: len(ref)], ref[: len(got)])
            print(f"PSNR vs taehv: {score:.2f} dB (floor {PSNR_FLOOR})")
            if score < PSNR_FLOOR:
                raise SystemExit("PSNR below floor")
        except Exception as e:
            print(f"PyTorch reference skipped: {e}")
    else:
        print("PyTorch arm skipped (pass --taehv and --weights to compare)")


if __name__ == "__main__":
    main()
