#!/usr/bin/env bash
# H3 tiny-decoder (Core ML / ANE) bridge smoke + optional A/B env doc.
#
# [1] C bridge opens the model and decodes frame 0 and a mid-clip frame (chunk alignment)
# [2] If zig-out/mlx-serve exists, prints the env vars for live linear vs TAE preview A/B
#
# Usage:
#   H3_TAE_MODEL=/path/to/minimax_h3_tae ./tests/test_h3_tae_preview.sh
#   H3_TAE_MODEL=/path/to/768.mlmodelc ./tests/test_h3_tae_preview.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${H3_TAE_MODEL:-}"
LAT_BIN="${H3_TAE_LATENT_BIN:-/tmp/h3-tae-probe/latent_export/video_latent_fp16.bin}"
LAT_META="${H3_TAE_LATENT_META:-/tmp/h3-tae-probe/latent_export/meta.json}"

if [[ -z "$MODEL" ]]; then
  if [[ -d /tmp/h3-tae-probe/dt-coreml/minimax_h3_tae ]]; then
    MODEL=/tmp/h3-tae-probe/dt-coreml/minimax_h3_tae
  elif [[ -d "/Applications/Draw Things.app/Contents/Resources/minimax_h3_tae" ]]; then
    echo "SKIP: set H3_TAE_MODEL to a scratch copy of the DT zip (never commit the asset)"
    exit 0
  else
    echo "SKIP: no H3_TAE_MODEL and no probe tree at /tmp/h3-tae-probe"
    exit 0
  fi
fi

[[ -f "$LAT_BIN" && -f "$LAT_META" ]] || {
  echo "SKIP: latent fixture missing ($LAT_BIN); run /tmp/h3-tae-probe/export_latent_bin.py first"
  exit 0
}

clang -c -fobjc-arc -O2 -Wno-deprecated-declarations -I"$ROOT/lib/h3_tae" \
  "$ROOT/lib/h3_tae/h3_tae_preview.m" -o /tmp/h3_tae_preview.o

cat >/tmp/h3_tae_test.c <<'EOF'
#include "h3_tae_preview.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int load_f32(const char *bin, int T, int H, int W, int C, float **out) {
  FILE *f = fopen(bin, "rb");
  if (!f) return -1;
  size_t need = (size_t)T * H * W * C * 2;
  unsigned char *fp16 = malloc(need);
  if (!fp16 || fread(fp16, 1, need, f) != need) { fclose(f); free(fp16); return -1; }
  fclose(f);
  float *lat = malloc((size_t)T * H * W * C * sizeof(float));
  if (!lat) { free(fp16); return -1; }
  for (size_t i = 0; i < (size_t)T * H * W * C; i++) {
    _Float16 h = 0;
    memcpy(&h, fp16 + i * 2, 2);
    lat[i] = (float)h;
  }
  free(fp16);
  *out = lat;
  return 0;
}

static int decode_idx(h3_tae_ctx *ctx, float *lat, int T, int H, int W, int C, int t_idx) {
  char err[512] = {0};
  int pw = 0, ph = 0;
  size_t rgb_len = (size_t)H * 16 * W * 16 * 3;
  uint8_t *rgb = calloc(1, rgb_len);
  if (!rgb) return -1;
  int rc = h3_tae_preview_decode_frame(ctx, lat, T, H, W, C, t_idx, rgb, &pw, &ph, err, sizeof err);
  if (rc != 0) { fprintf(stderr, "t_idx=%d: %s\n", t_idx, err); free(rgb); return rc; }
  unsigned sum = 0;
  for (size_t i = 0; i < 3000; i++) sum += rgb[i];
  printf("t_idx=%d ok %dx%d rgb_sum_sample=%u\n", t_idx, pw, ph, sum);
  free(rgb);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3) return 2;
  const char *model = argv[1];
  const char *lat_bin = argv[2];
  const int T = 22, H = 44, W = 44, C = 24;
  float *lat = NULL;
  if (load_f32(lat_bin, T, H, W, C, &lat) != 0) { perror("latent"); return 1; }
  char err[512] = {0};
  h3_tae_ctx *ctx = h3_tae_preview_open(model, 1, err, sizeof err);
  if (!ctx) { fprintf(stderr, "open: %s\n", err); return 1; }
  if (decode_idx(ctx, lat, T, H, W, C, 0) != 0) return 1;
  if (decode_idx(ctx, lat, T, H, W, C, 10) != 0) return 1;
  h3_tae_preview_close(ctx);
  free(lat);
  puts("bridge ok");
  return 0;
}
EOF

resolve_model() {
  if [[ "$MODEL" == *.mlmodelc ]]; then
    echo "$MODEL"
    return
  fi
  python3 - "$MODEL" <<'PY'
import json, sys, os
root = sys.argv[1]
meta = json.load(open(os.environ.get("H3_TAE_LATENT_META", "/tmp/h3-tae-probe/latent_export/meta.json")))
h, w = meta["h"], meta["w"]
px = max(h, w) * 16
for v in (768, 1024, 1280, 1536, 1792, 2048):
    if px <= v:
        print(os.path.join(root, f"{v}.mlmodelc"))
        break
else:
    print(os.path.join(root, "2048.mlmodelc"))
PY
}

ML="$(resolve_model)"
[[ -d "$ML" ]] || { echo "FAIL: no Core ML bundle at $ML"; exit 1; }

clang -O2 -I"$ROOT/lib/h3_tae" /tmp/h3_tae_test.c /tmp/h3_tae_preview.o -o /tmp/h3_tae_test \
  -framework CoreML -framework Foundation
/tmp/h3_tae_test "$ML" "$LAT_BIN"

echo ""
echo "Live preview A/B (after zig build):"
echo "  export MINIMAX_H3_TAE_MODEL=$MODEL"
echo "  export MINIMAX_H3_PREVIEW_PROF=1"
echo "  # TAE (ANE): default when MODEL set"
echo "  # Linear control: MINIMAX_H3_TAE=0"
