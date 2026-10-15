#!/usr/bin/env bash
# Pre-denoise pin parity: an in-request latent chain and a later
# latent_prefix_path extend must pin the same latent tail.
#
# Window 0 of the chain is also saved and compared to a standalone
# latent_out of the same seed. The pin files are written before that
# window's denoise steps (MLX_SERVE_H3_PIN_TRACE).
#
# Usage: [H3_MODEL=<dir>] ./tests/test_h3_pin_parity.sh [port]
# SKIP when the pack is absent. Uses --skip-mem-preflight because the
# staged bill is larger than a busy machine's free RAM.
set -euo pipefail
PORT="${1:-11367}"
MODEL="${H3_MODEL:-$HOME/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit}"
[ -f "$MODEL/transformer.safetensors" ] || { echo "SKIP: no MiniMax-H3 pack at $MODEL (set H3_MODEL)"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

TRACE="$(mktemp -d /tmp/h3_pin_parity.XXXXXX)"
TAKE="$TRACE/take.safetensors"
LOG=/tmp/test_h3_pin_parity.log
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

echo "chain (seed 7, overlap 22) ..."
post "$TRACE/chain.json" "$(python3 -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"num_frames":56,"width":64,"height":64,"steps":2,"seed":7,"chain_windows":2,"latent_prefix_frames":22}))' "$PROMPT")"

echo "standalone latent_out (same seed) ..."
post "$TRACE/take.json" "$(python3 -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"num_frames":56,"width":64,"height":64,"steps":2,"seed":7,"chain_windows":1,"latent_out":sys.argv[2]}))' "$PROMPT" "$TAKE")"

echo "extend from saved latent ..."
post "$TRACE/extend.json" "$(python3 -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"num_frames":56,"width":64,"height":64,"steps":2,"seed":9,"chain_windows":1,"latent_prefix_path":sys.argv[2],"latent_prefix_frames":22}))' "$PROMPT" "$TAKE")"

python3 - "$TRACE" <<'PY'
import hashlib, json, struct, sys
from pathlib import Path

trace = Path(sys.argv[1])
# overlap 22 → 7 video latent frames, 37 audio latent frames.
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

def tail(shape, raw, n, axis):
    # Row-major f32. A slice on an axis that is not last is strided, so the
    # pin is not a suffix of the parent buffer.
    import array
    vals = array.array("f")
    vals.frombytes(raw)
    if vals.itemsize != 4 or sys.byteorder != "little":
        raise SystemExit("need little-endian f32")
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    t = shape[axis]
    if t < n:
        raise SystemExit(f"shape {shape} shorter than pin {n}")
    lo = t - n
    out = array.array("f")

    def rec(dim, base):
        if dim == len(shape):
            out.append(vals[base])
            return
        span = range(lo, t) if dim == axis else range(shape[dim])
        for i in span:
            rec(dim + 1, base + i * strides[dim])

    rec(0, 0)
    return out.tobytes()

window0 = load(trace / "window0.safetensors")
take = load(trace / "take.safetensors")
pin_chain = load(trace / "pin-0.safetensors")
pin_ext = load(trace / "pin-1.safetensors")

fail = False
for key in ("video_latent", "audio_latent"):
    if window0[key][1] != take[key][1]:
        print(f"FAIL: window0 {key} != latent_out ({sha(window0[key][1])} vs {sha(take[key][1])})")
        fail = True
    else:
        print(f"PASS: window0 {key} == latent_out sha256={sha(window0[key][1])[:16]}")

v_tail = tail(window0["video_latent"][0], window0["video_latent"][1], PIN_V, 2)
a_tail = tail(window0["audio_latent"][0], window0["audio_latent"][1], PIN_A, 3)
if pin_chain["video_latent"][1] != v_tail or pin_chain["audio_latent"][1] != a_tail:
    print("FAIL: chain pin is not the tail of window 0")
    fail = True
else:
    print("PASS: chain pin is the tail of window 0")

if pin_chain["video_latent"][1] != pin_ext["video_latent"][1] or pin_chain["audio_latent"][1] != pin_ext["audio_latent"][1]:
    print(f"FAIL: chain pin != extend pin video {sha(pin_chain['video_latent'][1])[:16]} vs {sha(pin_ext['video_latent'][1])[:16]}")
    fail = True
else:
    print(f"PASS: chain pin == extend pin sha256={sha(pin_chain['video_latent'][1])[:16]}")

sys.exit(1 if fail else 0)
PY
echo "ALL PASS"
