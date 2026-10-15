#!/usr/bin/env bash
# Launch mlx-serve with in-place FL2VA hybrid (default b20-49 adaln merge).
# FL2VA-quality DiT + REF2VA later-block adaln_proj for character refs.
# Build pack: tests/merge_minimax_h3_hybrid.py
# Stock REF2VA (sheets / stills): ./start-minimax-h3-ref2va.sh
# FL2VA (first/last frame): ./start-minimax-h3.sh
#
# Quit ComfyUI first if memory preflight fails (~28 GB peak staged load for 8-bit).
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

pick_bin() {
  if [[ -n "${MLX_SERVE_BIN:-}" ]]; then
    echo "$MLX_SERVE_BIN"
    return
  fi
  if [[ -x "$REPO/zig-out/bin/mlx-serve" ]]; then
    echo "$REPO/zig-out/bin/mlx-serve"
    return
  fi
  if command -v mlx-serve >/dev/null 2>&1; then
    command -v mlx-serve
    return
  fi
  local app="/Applications/MLX Core.app/Contents/MacOS/mlx-serve"
  if [[ -x "$app" ]]; then
    echo "$app"
    return
  fi
  return 1
}

BIN="$(pick_bin)" || {
  echo "mlx-serve binary not found" >&2
  echo "Install v26.8.3+ (brew upgrade mlx-serve / update MLX Core) or:" >&2
  echo "  zig build -Doptimize=ReleaseFast" >&2
  exit 1
}
MODEL="${MINIMAX_H3_MODEL:-$HOME/Models/omlx/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit}"
PORT="${MLX_SERVE_PORT:-11361}"

if [[ ! -x "$BIN" ]]; then
  echo "mlx-serve binary not executable: $BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL/transformer.safetensors" ]]; then
  echo "Hybrid pack not found at $MODEL" >&2
  echo "Build it with:" >&2
  echo "  uv run --with safetensors --with numpy tests/merge_minimax_h3_hybrid.py" >&2
  exit 1
fi

echo "Using binary: $BIN" >&2
echo "Hybrid model: $MODEL" >&2
exec "$BIN" \
  --model "$MODEL" \
  --serve \
  --host 0.0.0.0 \
  --port "$PORT" \
  --skip-mem-preflight
