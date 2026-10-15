#!/usr/bin/env bash
# Chained windows + latent_out: request accepted; saved sidecar is the last
# window. With MLX_SERVE_H3_PIN_TRACE, the last-window sidecar's pinned head
# matches window 0's latent tail (overlap region), but the full tensors differ.
#
# Usage: [H3_MODEL=<dir>] ./tests/test_h3_chain_latent_out.sh [port]
set -euo pipefail
PORT="${1:-11368}"
MODEL="${H3_MODEL:-$HOME/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit}"
[ -f "$MODEL/transformer.safetensors" ] || { echo "SKIP: no MiniMax-H3 pack at $MODEL (set H3_MODEL)"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

TRACE="$(mktemp -d /tmp/h3_chain_latent_out.XXXXXX)"
SIDE="$TRACE/chain_last.safetensors"
LOG=/tmp/test_h3_chain_latent_out.log
PROMPT='a calico cat on a sill. overall_soundscape: soft rain.'

MLX_SERVE_H3_PIN_TRACE="$TRACE" "$BIN" --skip-mem-preflight --model "$MODEL" --serve --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -rf "$TRACE"' EXIT
for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: server did not start"; tail -12 "$LOG"; exit 1; }
  sleep 1
done

post() {
  local out="$1" body="$2"
  local code
  code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" \
    -H 'Content-Type: application/json' -d "$body" -o "$out" -w "%{http_code}")
  if [ "$code" != "200" ]; then
    echo "FAIL: http $code"
    head -c 300 "$out"; echo
    tail -20 "$LOG"
    exit 1
  fi
}

echo "chain + latent_out (2 windows, overlap 22) ..."
post "$TRACE/resp.json" "$(python3 -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"num_frames":56,"width":64,"height":64,"steps":2,"seed":7,"chain_windows":2,"latent_prefix_frames":22,"latent_out":sys.argv[2]}))' "$PROMPT" "$SIDE")"

[ -f "$SIDE" ] || { echo "FAIL: latent_out file not written"; exit 1; }
echo "PASS: chain latent_out accepted and file written"

python3 - "$TRACE" "$SIDE" <<'PY'
import hashlib, json, struct, sys
from pathlib import Path

trace = Path(sys.argv[1])
side_path = Path(sys.argv[2])
PIN_V, PIN_A = 7, 37

def load(path):
    data = Path(path).read_bytes()
    n = struct.unpack_from("<Q", data, 0)[0]
    header = json.loads(data[8:8 + n])
    base = 8 + n
    out = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[key] = (meta["shape"], data[base + start:base + end])
    return out

def sha(b):
    return hashlib.sha256(b).hexdigest()

def slice_axis(shape, raw, axis, lo, hi):
    import array
    vals = array.array("f")
    vals.frombytes(raw)
    if vals.itemsize != 4 or sys.byteorder != "little":
        raise SystemExit("need little-endian f32")
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    t = shape[axis]
    if hi > t:
        raise SystemExit(f"slice [{lo}:{hi}) past axis {axis} len {t}")
    out = array.array("f")

    def rec(dim, base):
        if dim == len(shape):
            out.append(vals[base])
            return
        span = range(lo, hi) if dim == axis else range(shape[dim])
        for i in span:
            rec(dim + 1, base + i * strides[dim])

    rec(0, 0)
    return out.tobytes()

def head(shape, raw, n, axis):
    return slice_axis(shape, raw, axis, 0, n)

def tail(shape, raw, n, axis):
    t = shape[axis]
    return slice_axis(shape, raw, axis, t - n, t)

window0 = load(trace / "window0.safetensors")
side = load(side_path)

fail = False
v_head = head(side["video_latent"][0], side["video_latent"][1], PIN_V, 2)
v_tail0 = tail(window0["video_latent"][0], window0["video_latent"][1], PIN_V, 2)
a_head = head(side["audio_latent"][0], side["audio_latent"][1], PIN_A, 3)
a_tail0 = tail(window0["audio_latent"][0], window0["audio_latent"][1], PIN_A, 3)

if v_head != v_tail0 or a_head != a_tail0:
    print("FAIL: last-window sidecar head != window0 tail (pin region)")
    fail = True
else:
    print("PASS: sidecar pin head matches window0 latent tail")

for key in ("video_latent", "audio_latent"):
    if side[key][1] == window0[key][1]:
        print(f"FAIL: full sidecar {key} equals window0 (must be last window only)")
        fail = True
    else:
        print(f"PASS: full sidecar {key} != window0 ({sha(side[key][1])[:16]} vs {sha(window0[key][1])[:16]})")

sys.exit(1 if fail else 0)
PY
echo "ALL PASS"
