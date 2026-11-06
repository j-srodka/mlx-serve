#!/usr/bin/env bash
# H3 tiny-decoder (Core ML / ANE) bridge smoke + optional A/B env doc.
#
# [1] C bridge opens the model and decodes frame 0 and a mid-clip frame (chunk alignment)
# [2] If zig-out/mlx-serve exists, prints the env vars for live linear vs TAE preview A/B
#
# Usage:
#   ./tests/test_h3_tae_preview.sh
#   H3_TAE_MODEL=~/.mlx-serve/shared/minimax_h3_tae ./tests/test_h3_tae_preview.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHARED="${MINIMAX_H3_TAE_DIR:-$HOME/.mlx-serve/shared/minimax_h3_tae}"
MODEL="${H3_TAE_MODEL:-}"
LAT_BIN="${H3_TAE_LATENT_BIN:-/tmp/h3-tae-probe/latent_export/video_latent_fp16.bin}"
LAT_META="${H3_TAE_LATENT_META:-/tmp/h3-tae-probe/latent_export/meta.json}"

if [[ "$(uname -s)" != Darwin ]]; then
  echo "SKIP: Core ML taeh3 bridge is macOS-only"
  exit 0
fi

if [[ -z "$MODEL" ]]; then
  if [[ -d "$SHARED/768.mlmodelc" ]]; then
    MODEL="$SHARED"
  elif [[ -d "$ROOT/.cache/minimax_h3_tae/768.mlmodelc" ]]; then
    MODEL="$ROOT/.cache/minimax_h3_tae"
  else
    echo "SKIP: no MIT Core ML bundle — run scripts/fetch_minimax_h3_tae.sh or set H3_TAE_MODEL"
    exit 0
  fi
fi

clang -c -fobjc-arc -O2 -Wno-deprecated-declarations -I"$ROOT/lib/h3_tae" \
  "$ROOT/lib/h3_tae/h3_tae_preview.m" -o /tmp/h3_tae_preview.o

# Open-only: dangling weights/weight.bin (import from a temp probe) must be replaced.
cat >/tmp/h3_tae_open.c <<'EOF'
#include "h3_tae_preview.h"
#include <stdio.h>
int main(int argc, char **argv) {
  if (argc < 2) return 2;
  char err[512] = {0};
  h3_tae_ctx *ctx = h3_tae_preview_open(argv[1], 1, err, sizeof err);
  if (!ctx) { fprintf(stderr, "open: %s\n", err); return 1; }
  h3_tae_preview_close(ctx);
  return 0;
}
EOF
clang -O2 -I"$ROOT/lib/h3_tae" /tmp/h3_tae_open.c /tmp/h3_tae_preview.o -o /tmp/h3_tae_open \
  -framework CoreML -framework Foundation
ML768="$MODEL"
if [[ "$MODEL" != *.mlmodelc ]]; then ML768="$MODEL/768.mlmodelc"; fi
WB_SRC="$(dirname "$ML768")/weight.bin"
[[ -f "$WB_SRC" ]] || WB_SRC="$MODEL/weight.bin"
DANGLE=$(mktemp -d /tmp/h3_tae_dangle.XXXX)
trap 'rm -rf "$DANGLE"' EXIT
ln "$WB_SRC" "$DANGLE/weight.bin" 2>/dev/null || cp "$WB_SRC" "$DANGLE/weight.bin"
rsync -a --exclude 'weights/weight.bin' "$ML768/" "$DANGLE/768.mlmodelc/"
mkdir -p "$DANGLE/768.mlmodelc/weights"
ln -s /nonexistent/h3-tae-probe/weight.bin "$DANGLE/768.mlmodelc/weights/weight.bin"
/tmp/h3_tae_open "$DANGLE/768.mlmodelc"
WB="$DANGLE/768.mlmodelc/weights/weight.bin"
if [[ -L "$WB" ]] || [[ ! -f "$WB" ]]; then
  echo "FAIL: dangling weight.bin not repaired ($WB)"
  exit 1
fi
echo "dangling weight.bin repaired"

[[ -f "$LAT_BIN" && -f "$LAT_META" ]] || {
  echo "SKIP: latent fixture missing ($LAT_BIN); see tests/fixtures/h3_tae/README.md"
  exit 0
}

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

static int stripe_dark_column_frac(const uint8_t *rgb, int pw, int ph) {
  /* Dense mis-read of strided Core ML output: mostly dark columns with sparse bright ones. */
  int dark_cols = 0;
  for (int x = 0; x < pw; x++) {
    unsigned long sum = 0;
    for (int y = 0; y < ph; y++) {
      const uint8_t *p = rgb + (y * pw + x) * 3;
      sum += (unsigned)(p[0] + p[1] + p[2]) / 3;
    }
    const unsigned mean = (unsigned)(sum / (unsigned)ph);
    if (mean < 8) dark_cols++;
  }
  return (dark_cols * 100) / pw;
}

static void maybe_dump_ppm(const uint8_t *rgb, int pw, int ph, int t_idx) {
  const char *path = getenv("H3_TAE_DUMP_PPM");
  if (!path || !path[0]) return;
  FILE *f = fopen(path, "wb");
  if (!f) return;
  fprintf(f, "P6\n%d %d\n255\n", pw, ph);
  fwrite(rgb, 1, (size_t)pw * ph * 3, f);
  fclose(f);
  fprintf(stderr, "wrote ppm t_idx=%d -> %s\n", t_idx, path);
}

static int decode_idx(h3_tae_ctx *ctx, float *lat, int T, int H, int W, int C, int t_idx, int check_stripe) {
  char err[512] = {0};
  int pw = 0, ph = 0;
  size_t rgb_len = (size_t)H * 16 * W * 16 * 3;
  uint8_t *rgb = calloc(1, rgb_len);
  if (!rgb) return -1;
  int rc = h3_tae_preview_decode_frame(ctx, lat, T, H, W, C, t_idx, rgb, &pw, &ph, err, sizeof err);
  if (rc != 0) { fprintf(stderr, "t_idx=%d: %s\n", t_idx, err); free(rgb); return rc; }
  if (check_stripe) {
    const int dark_pct = stripe_dark_column_frac(rgb, pw, ph);
    if (dark_pct >= 30) {
      fprintf(stderr, "FAIL: stripe guard dark_columns=%d%% (want <30%%)\n", dark_pct);
      free(rgb);
      return 1;
    }
    printf("stripe guard ok dark_columns=%d%% (<%d)\n", dark_pct, 30);
  }
  maybe_dump_ppm(rgb, pw, ph, t_idx);
  printf("t_idx=%d ok %dx%d\n", t_idx, pw, ph);
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
  if (decode_idx(ctx, lat, T, H, W, C, 0, 1) != 0) return 1;
  if (decode_idx(ctx, lat, T, H, W, C, 10, 0) != 0) return 1;
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
echo "Live preview A/B (after zig build -Doptimize=ReleaseFast):"
echo "  # TAE default when bundle is in pack minimax_h3_tae/ or $SHARED"
echo "  export MINIMAX_H3_PREVIEW_PROF=1"
echo "  # Linear control:"
echo "  export MINIMAX_H3_TAE=0"
echo "  # Explicit override:"
echo "  export MINIMAX_H3_TAE_MODEL=$MODEL"
