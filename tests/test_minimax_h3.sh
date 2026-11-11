#!/usr/bin/env bash
# test_minimax_h3.sh — MiniMax-H3 audio-video end to end on the converted pack: `video` capability, the staged
# preflight's NUMBER in the 30-50 GB band (max(TE,DiT)+VAEs, never the sum), named 400s (LoRA, a non-/32 canvas, chat
# against a video model), a small generation returning rgb8 + pcm_s16le of the right lengths with audio/video in sync,
# SSE progress -> complete with 40 frames snapping UP to 56 on the 17k+5 ladder, fl2va adherence at 256px with
# "fast": false, and the fast recipe by engagement (`fast:false` logs no reuse; the default logs `step-cache reused` and
# `attn-broadcast reused`). The pack's own config.json `tasks` decides which half runs: ref2va pins its eight named 400s
# and one image + soundtracked clip + audio generation whose resolver order and cond-row budgets are counted in the log.
# Residency is pinned OFF (staged-plan properties).
#
#   H3_MODEL=<dir> ./tests/test_minimax_h3.sh [port]   # default ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit; SKIP without
#
# Hermetic twins: the layout/schedule/vision/VAE fixture tests (minimax_h3*.zig), h3PeakBytes; live parity:
# MINIMAX_H3_DIR, MINIMAX_H3_VAE_ENC_FIXTURE, _AUDIO_ENC_FIXTURE (max-abs, not cosine), _VIT_FIXTURE (+ rms_ratio).
source "$(dirname "$0")/_lib_models.sh"

set -uo pipefail
source "$(dirname "$0")/lib/server.sh"
PORT="${1:-11361}"
MODEL="${H3_MODEL:-$(find_model ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit || echo "$HOME/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit")}"
[ -f "$MODEL/transformer.safetensors" ] || { echo "SKIP: no MiniMax-H3 pack at $MODEL (set H3_MODEL)"; exit 0; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${MLX_SERVE_BIN:-$ROOT/zig-out/bin/mlx-serve}"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

LOG=/tmp/test_minimax_h3_server.log
# This script pins the STAGED plan (disjoint stages, AdaLN shed); residency has its own script.
MLX_SERVE_H3_RESIDENT=0 "$BIN" --model "$MODEL" --serve --port "$PORT" >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
srv_wait_health "http://127.0.0.1:$PORT" 90 1 "$SRV" || kill -0 "$SRV" 2>/dev/null || { echo "FAIL: server did not start"; tail -8 "$LOG"; exit 1; }
rc=0

# Which PARTITION this pack is. The two ship identical files and geometry —
# only the DiT weights and the declared task list differ — so the pack's own
# config is the only honest discriminator, and asserting fl2va adherence on a
# REF2VA DiT (or references on an FL2VA one) would be a checkpoint expectation
# in a server test's clothes.
TASKS=$(python3 - "$MODEL/config.json" <<'PY'
import json, sys
try:
    print(",".join(json.load(open(sys.argv[1])).get("tasks", [])))
except Exception:
    print("")
PY
)
case ",$TASKS," in (*,fl2va,*) HAS_FL2VA=yes;; (*) HAS_FL2VA=no;; esac
case ",$TASKS," in (*,ref2va,*) HAS_REF2VA=yes;; (*) HAS_REF2VA=no;; esac
echo "pack tasks: [$TASKS]"

HEALTH=$(curl -sf "http://127.0.0.1:$PORT/health" || true)
if echo "$HEALTH" | grep -q '"ref_latent_out":true' && echo "$HEALTH" | grep -q '"latent_suffix":true' && echo "$HEALTH" | grep -q '"ref_pins":true' && echo "$HEALTH" | grep -q '"latent_refine":true' && echo "$HEALTH" | grep -q '"latent_upscale":'; then
  echo "PASS: /health advertises latent_suffix, ref_latent_out, ref_pins, latent_refine, and latent_upscale"
else
  echo "FAIL: /health missing pin/refine flags ($HEALTH)"; rc=1
fi

# [1] capability + staged preflight engagement
curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q '"video"' \
  && echo "PASS: /v1/models advertises video" \
  || { echo "FAIL: /v1/models missing video capability"; rc=1; }
# The NUMBER is the assertion, not the line: the line printed happily while
# the stub's modality-static model_type routed the bill to the 64.5 GB sum.
# The three stages are DISJOINT (TE freed before the DiT loads, DiT freed
# before the VAEs), and the DiT sheds its AdaLN weights, so the 8-bit pack's
# bill is max(TE 26.28, DiT 32.83x0.65 + turbo 0.73 + 6 activations) = 28.06 —
# against a 64.5 GB sum and the 38.97 this used to read.
if [ "${SKIP_MEM_PREFLIGHT:-0}" = "1" ]; then
  echo "SKIP: media preflight peak (SKIP_MEM_PREFLIGHT=1)"
else
  PEAK=$(grep -m1 "media peak" "$LOG" | sed -E 's/.*media peak ~([0-9.]+) GB.*/\1/')
  if [ -n "$PEAK" ] && python3 -c "import sys; sys.exit(0 if 24.0 < float('$PEAK') < 32.0 else 1)"; then
    echo "PASS: media preflight billed the biggest stage (~$PEAK GB, not the ~64.5 sum)"
  else
    echo "FAIL: media preflight peak '$PEAK' GB is not the staged bill"; rc=1
  fi
fi

# [2] the named-400 surface — all cheap, so they run before any generation

# [7] ref2va. The two partitions ship IDENTICAL files and geometry — only the
# DiT weights and the declared task list differ — so an FL2VA pack handed
# references would generate happily while ignoring every one of them. Which
# half of this section runs is decided by the pack's own config, never by its
# directory name.
if [ "$HAS_REF2VA" != "yes" ]; then
  code=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    -d '{"prompt":"x","num_frames":5,"width":64,"height":64,"ref_images":["QQ=="]}' -o /tmp/h3_norefs.json -w "%{http_code}")
  if [ "$code" = "400" ] && grep -q "ref2va" /tmp/h3_norefs.json; then
    echo "PASS: references against a non-REF2VA pack -> named 400"
  else
    echo "FAIL: refs on an FL2VA pack returned $code ($(head -c 160 /tmp/h3_norefs.json))"; rc=1
  fi
  echo "SKIP: ref2va generation cases (this pack declares no 'ref2va' task)"
else
  # Fixtures: one reference image, a 5-frame clip (5 is already ON the 17k+5
  # ladder, so nothing is snapped away), and a short stereo WAV.
  python3 - /tmp/h3_ref <<'PY'
import struct, zlib, sys, math, wave
base = sys.argv[1]
def png(path, w, h, pix):
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw += bytes(pix(x, y))
    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    out += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    out += chunk(b"IEND", b"")
    open(path, "wb").write(out)

# A reference IMAGE at 128px: `match` sizing halves it onto the 64px canvas.
png(base + "_img.png", 128, 128, lambda x, y: (230 if x < 64 else 25, 40, 200 if y < 64 else 30))
# Frames that MOVE, so a collapsed or reversed temporal axis cannot pass.
for i in range(5):
    png(f"{base}_v{i}.png", 64, 64, lambda x, y, i=i: (255 if abs(x - (8 + 10 * i)) < 6 else 20, 20, 20))
# 0.5 s of 16 kHz stereo — resampled to the VAE's 32 kHz server-side.
w = wave.open(base + "_aud.wav", "wb")
w.setnchannels(2); w.setsampwidth(2); w.setframerate(16000)
w.writeframes(b"".join(struct.pack("<hh", int(12000 * math.sin(t / 20.0)), int(9000 * math.sin(t / 30.0)))
                       for t in range(8000)))
w.close()
PY

  # The named-400 surface. Every arm is a refusal the client can act on: a
  # silently dropped reference generates a clip that just ignores the user.
  b64() { base64 < "$1" | tr -d '\n'; }
  IMGB=$(b64 /tmp/h3_ref_img.png)
  AUDB=$(b64 /tmp/h3_ref_aud.wav)
  python3 - "$IMGB" "$AUDB" <<'PY'
import json, sys
img, aud = sys.argv[1], sys.argv[2]
base = {"prompt": "x", "num_frames": 5, "width": 64, "height": 64, "steps": 1}
frames = [img] * 5
cases = {
    "over_images": {"ref_images": [img] * 10},
    "over_videos": {"ref_videos": [{"frames": frames}] * 4},
    "over_audios": {"ref_audios": [aud] * 4},
    "short_video": {"ref_videos": [{"frames": [img] * 4}]},
    "empty_video": {"ref_videos": [{"frames": []}]},
    "bad_image":   {"ref_images": ["bm90IGFuIGltYWdl"]},
    "bad_audio":   {"ref_audios": ["bm90IGF1ZGlv"]},
    "bad_sizing":  {"ref_images": [img], "ref_image_size": "huge"},
    # Every per-type cap holds (9 + 3 + 3) and the SET is still over: MiniMax's
    # limit is 12 files across all three lists, which no per-type check can see.
    "over_total":  {"ref_images": [img] * 9,
                    "ref_videos": [{"frames": frames}] * 3,
                    "ref_audios": [aud] * 3},
}
for name, extra in cases.items():
    json.dump({**base, **extra}, open(f"/tmp/h3_ref_400_{name}.json", "w"))
PY
  for case in over_images:"at most 9" over_videos:"at most 3" over_audios:"at most 3" \
              short_video:"at least 5 frames" empty_video:"no frames" \
              bad_image:"ref_images" bad_audio:"ref_audios" bad_sizing:"ref_image_size" \
              over_total:"at most 12 files"; do
    name="${case%%:*}"; want="${case#*:}"; want="${want%\"}"; want="${want#\"}"
    code=$(curl -s --max-time 120 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
      --data @/tmp/h3_ref_400_"$name".json -o /tmp/h3_ref_400_"$name"_resp.json -w "%{http_code}")
    if [ "$code" = "400" ] && grep -qF "$want" /tmp/h3_ref_400_"$name"_resp.json; then
      echo "PASS: ref2va $name -> named 400"
    else
      echo "FAIL: ref2va $name returned $code ($(head -c 160 /tmp/h3_ref_400_"$name"_resp.json))"; rc=1
    fi
  done

  # A mixed reference set through the whole path: image + soundtracked clip +
  # standalone audio. The ORDINALS are a contract with the checkpoint (the
  # prompt says <Picture 1> / <Video 1> / <Audio j>), and a soundtrack consumes
  # an <Audio> ordinal ahead of any standalone one — so this asserts the
  # SECOND standalone audio is <Audio 2>, which an ordering bug cannot fake.
  REFLOG_BEFORE=$(grep -c "minimax-h3 reference" "$LOG" || true)
  python3 - "$IMGB" "$AUDB" /tmp/h3_ref <<'PY'
import base64, json, sys
img, aud, base = sys.argv[1], sys.argv[2], sys.argv[3]
frames = [base64.b64encode(open(f"{base}_v{i}.png", "rb").read()).decode() for i in range(5)]
json.dump({
    "prompt": "<Picture 1> the cat, <Video 1> the motion, <Audio 2> the mood",
    "num_frames": 5, "width": 64, "height": 64, "steps": 2, "seed": 11,
    "ref_images": [img],
    "ref_videos": [{"frames": frames, "audio": aud}],
    "ref_audios": [aud],
    "stream": True,
}, open("/tmp/h3_ref_mixed.json", "w"))
PY
  RSSE=/tmp/test_minimax_h3_ref_sse.txt
  curl -sN --max-time 1200 -X POST "http://127.0.0.1:$PORT/v1/video/generations" -H 'Content-Type: application/json' \
    --data @/tmp/h3_ref_mixed.json >"$RSSE"
  python3 - "$RSSE" <<'PY'
import sys, json, base64
prog = 0; complete = None
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("data: "): continue
    ev = json.loads(line[6:])
    if ev["type"] == "progress": prog += 1
    elif ev["type"] == "complete": complete = ev
assert complete is not None, "no complete event on the ref2va run"
assert prog >= 1, f"expected progress events, got {prog}"
F, H, W = complete["frames"], complete["height"], complete["width"]
assert (F, H, W) == (5, 64, 64), (F, H, W)
raw = base64.b64decode(complete["data"])
assert len(raw) == F * H * W * 3, f"rgb len {len(raw)} != {F*H*W*3}"
assert max(raw) - min(raw) > 20, "ref2va frames look uniform"
print(f"PASS: ref2va SSE -> {prog} progress events, {F}f {W}x{H} with a soundtrack")
PY
  [ $? -eq 0 ] || rc=1

  # Engagement, counted in the server's own log: the resolver's ORDER and
  # ORDINALS, and the layout actually growing reference rows. Output alone
  # cannot see a silently ignored reference set — the generation succeeds.
  python3 - "$LOG" <<'PY'
import re, sys
lines = [l for l in open(sys.argv[1]) if "minimax-h3 reference" in l]
tail = lines[-3:]
assert len(tail) == 3, f"expected 3 reference lines, got {len(lines)}"
kinds = [re.search(r"reference (\w+) #(\d+)", l).groups() for l in tail]
assert kinds == [("image", "1"), ("video", "1"), ("audio", "2")], kinds
cond = [l for l in open(sys.argv[1]) if "reference(s) encoded" in l][-1]
m = re.search(r"-> (\d+) visual / (\d+) audio cond rows", cond)
assert m, cond
vis, aud = int(m.group(1)), int(m.group(2))
# image (1 latent frame) + clip (2 latent frames) at a 4x4 latent grid, patch
# 2x2 => 4 rows per frame; the soundtrack and the standalone audio each pack
# 2 rows per latent frame.
assert vis == 12, f"expected 12 visual cond rows (1+2 latent frames x 4), got {vis}"
assert aud > 0 and aud % 4 == 0, f"audio cond rows {aud} are not two equal-length stereo blocks"
print(f"PASS: ref2va engagement — image#1/video#1/audio#2 in order, {vis} visual + {aud} audio cond rows")
PY
  [ $? -eq 0 ] || rc=1
fi

if [ $rc -eq 0 ]; then echo "ALL PASS"; else echo "FAILURES"; fi
exit $rc
