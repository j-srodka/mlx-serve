#!/usr/bin/env bash
# Launch mlx-serve with MiniMax-H3 REF2VA 8-bit (ddalcu pack).
# Needs mlx-serve ≥ v26.8.3 (ref_images API). Prefer MLX_SERVE_BIN override,
# then zig-out build, then Homebrew / MLX Core app binary.
# FL2VA (first/last frame): ./start-minimax-h3.sh
#
# Quit ComfyUI first if memory preflight fails (~28 GB peak staged load for 8-bit).
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
# Run from repo so @rpath finds lib/llama and lib/mlx dylibs when using zig-out.
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
MODEL="${MINIMAX_H3_MODEL:-}"
PORT="${MLX_SERVE_PORT:-11361}"

# MODEL ROOTS, in try-order. This used to hardcode the default to ~/.mlx-serve/models/ddalcu/
# — but that tree holds no ddalcu models at all (only Falconsai), while the downloaded H3
# REF2VA weights live in ~/Models/omlx/ddalcu/. So a bare ./start-minimax-h3-ref2va.sh died
# with "Model not found" and only worked with a MINIMAX_H3_MODEL override (hit 2026-09-26
# starting the pack extract). Try the known roots and take the first that actually holds the
# weights; an explicit MINIMAX_H3_MODEL override still wins.
_MODEL_ROOTS=("$HOME/.mlx-serve/models/ddalcu" "$HOME/Models/omlx/ddalcu")
if [[ -z "$MODEL" ]]; then
  for _root in "${_MODEL_ROOTS[@]}"; do
    if [[ -f "$_root/MiniMax-H3-REF2VA-MLX-Serve-8bit/transformer.safetensors" ]]; then
      MODEL="$_root/MiniMax-H3-REF2VA-MLX-Serve-8bit"
      break
    fi
  done
fi

if [[ ! -x "$BIN" ]]; then
  echo "mlx-serve binary not executable: $BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL/transformer.safetensors" ]]; then
  echo "Model not found (tried: ${_MODEL_ROOTS[*]/%//MiniMax-H3-REF2VA-MLX-Serve-8bit}${MODEL:+ ; override: $MODEL})" >&2
  if [[ -f "$MODEL/transformer.safetensors.partial" ]]; then
    echo "Download still incomplete (transformer.safetensors.partial present)." >&2
  fi
  exit 1
fi

echo "Using binary: $BIN" >&2
exec "$BIN" \
  --model "$MODEL" \
  --serve \
  --host 0.0.0.0 \
  --port "$PORT" \
  --skip-mem-preflight
