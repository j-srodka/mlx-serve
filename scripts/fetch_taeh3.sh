#!/usr/bin/env bash
# Fetch MIT taeh3 decoder weights (madebyollin/taehv) for Core ML conversion.
# Does NOT download Draw Things' minimax_h3_tae.zip.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${TAEH3_DIR:-$ROOT/.cache/taeh3}"
URL="${TAEH3_URL:-https://huggingface.co/OzzyGT/taeh3/resolve/main/taeh3.safetensors}"
SHA="${TAEH3_SHA256:-4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13}"
FILE="$DEST/taeh3.safetensors"

mkdir -p "$DEST"
if [[ -f "$FILE" ]]; then
  got=$(shasum -a 256 "$FILE" | awk '{print $1}')
  if [[ "$got" == "$SHA" ]]; then
    echo "[fetch-taeh3] ok $FILE"
    exit 0
  fi
  echo "[fetch-taeh3] sha mismatch ($got) — refetching"
fi
echo "[fetch-taeh3] downloading $URL"
curl -fsSL -o "$FILE.part" "$URL"
mv "$FILE.part" "$FILE"
got=$(shasum -a 256 "$FILE" | awk '{print $1}')
[[ "$got" == "$SHA" ]] || { echo "[fetch-taeh3] FAIL sha $got want $SHA"; exit 1; }
echo "[fetch-taeh3] wrote $FILE"
echo "[fetch-taeh3] Next: coremltools stateful chunk-2 export (see handoff-h3-tae-preview.md Phase 1)"
