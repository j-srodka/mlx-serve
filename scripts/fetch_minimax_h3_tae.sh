#!/usr/bin/env bash
# Install MiniMax-H3 Core ML taeh3 preview bundles (MIT-derived layout).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${MINIMAX_H3_TAE_DIR:-$HOME/.mlx-serve/shared/minimax_h3_tae}"
OUT_PARENT="$(dirname "$OUT")"
HF_REPO="${MINIMAX_H3_TAE_HF:-}"

link_variant_weights() {
  local root="$1"
  [[ -f "$root/weight.bin" ]] || return 0
  local v d dst
  for v in 768 1024 1280 1536 1792 2048; do
    d="$root/${v}.mlmodelc/weights"
    mkdir -p "$d"
    dst="$d/weight.bin"
    if [[ -L "$dst" ]] || [[ ! -f "$dst" ]]; then
      if ! rm -f "$dst"; then
        echo "[fetch-minimax-h3-tae] cannot replace $dst" >&2
        continue
      fi
      ln "$root/weight.bin" "$dst" 2>/dev/null || cp "$root/weight.bin" "$dst"
    fi
  done
}

if [[ -d "$OUT/768.mlmodelc" && -f "$OUT/weight.bin" ]]; then
  link_variant_weights "$OUT"
  echo "[fetch-minimax-h3-tae] ok $OUT"
  exit 0
fi

mkdir -p "$OUT"

if [[ -n "$HF_REPO" ]]; then
  echo "[fetch-minimax-h3-tae] downloading $HF_REPO -> $OUT"
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$HF_REPO" --local-dir "$OUT" --local-dir-use-symlinks False
  elif command -v hf >/dev/null 2>&1; then
    hf download "$HF_REPO" --local-dir "$OUT"
  else
    echo "[fetch-minimax-h3-tae] install huggingface-cli or set MINIMAX_H3_TAE_IMPORT to a local tree" >&2
    exit 1
  fi
elif [[ -n "${MINIMAX_H3_TAE_IMPORT:-}" ]]; then
  echo "[fetch-minimax-h3-tae] importing ${MINIMAX_H3_TAE_IMPORT} -> $OUT"
  rsync -a --delete "${MINIMAX_H3_TAE_IMPORT}/" "$OUT/"
elif [[ -d "$ROOT/.cache/minimax_h3_tae/768.mlmodelc" ]]; then
  echo "[fetch-minimax-h3-tae] using $ROOT/.cache/minimax_h3_tae"
  rsync -a "$ROOT/.cache/minimax_h3_tae/" "$OUT/"
else
  echo "[fetch-minimax-h3-tae] running local convert (requires macOS + coremltools + torch)"
  "$ROOT/scripts/fetch_taeh3.sh"
  python3 "$ROOT/scripts/convert_taeh3_coreml.py" --out "$OUT_PARENT"
fi

[[ -d "$OUT/768.mlmodelc" ]] || { echo "[fetch-minimax-h3-tae] FAIL: missing 768.mlmodelc under $OUT" >&2; exit 1; }
link_variant_weights "$OUT"
echo "[fetch-minimax-h3-tae] installed $OUT"
echo "Optional: symlink into an H3 pack: ln -sf \"$OUT\" \"\$PACK/minimax_h3_tae\""
