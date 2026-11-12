#!/usr/bin/env bash
# Three-arm H3 preview comparison on one canvas:
#   A) preview OFF (baseline)
#   B) preview ON, linear latent2rgb (MINIMAX_H3_TAE=0)
#   C) preview ON, Core ML TAE on ANE (bundle in pack dir, shared cache, or MINIMAX_H3_TAE_MODEL)
#
# Usage:
#   ./tests/run_ab_tae.sh
#   MINIMAX_H3_TAE_MODEL=/path/to/minimax_h3_tae MLX_SERVE_PORT=11362 ./tests/run_ab_tae.sh
#
# Artifacts: $OUT/report_tae_ab.json, TAE_AB_REPORT.md, log_*.txt, sse_*.txt
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/zig-out/bin/mlx-serve"
MODEL="${MINIMAX_H3_MODEL:-$HOME/Models/omlx/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit}"
TAE_MODEL="${MINIMAX_H3_TAE_MODEL:-$HOME/.mlx-serve/shared/minimax_h3_tae}"
PORT="${MLX_SERVE_PORT:-11362}"
LOG="$HOME/.mlx-serve/logs/mlx-serve-${PORT}.log"
OUT="${H3_PREVIEW_AB_OUT:-/tmp/h3-preview-ab}"
mkdir -p "$OUT"

if [[ ! -x "$BIN" ]]; then
  echo "FAIL: missing $BIN — zig build -Doptimize=ReleaseFast in $REPO" >&2
  exit 1
fi
if [[ ! -d "$TAE_MODEL" ]] && [[ ! "$TAE_MODEL" == *.mlmodelc ]]; then
  echo "FAIL: MINIMAX_H3_TAE_MODEL not found: $TAE_MODEL" >&2
  echo "Run: $REPO/scripts/fetch_minimax_h3_tae.sh" >&2
  exit 1
fi
if [[ ! -f "$MODEL/transformer.safetensors" ]]; then
  echo "SKIP: H3 pack missing at $MODEL (set MINIMAX_H3_MODEL)" >&2
  exit 0
fi

SRV=""
stop_server() {
  if [[ -n "${SRV:-}" ]]; then
    kill "$SRV" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
    SRV=""
  fi
  pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
  pkill -f "zig-out/bin/mlx-serve.*--port $PORT" 2>/dev/null || true
  sleep 3
}
trap stop_server EXIT

start_server() {
  stop_server
  if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "FAIL: port $PORT still in use" >&2
    lsof -iTCP:"$PORT" -sTCP:LISTEN >&2 || true
    exit 1
  fi
  : >"$LOG"
  # shellcheck disable=SC2068
  env "$@" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
    --skip-mem-preflight --log-level info --timeout 0 >>"$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "FAIL: server did not start (see $LOG)" >&2
  tail -20 "$LOG" >&2
  exit 1
}

run_job() {
  local tag="$1"
  local preview_flag="$2"
  local body
  body=$(python3 -c "import json; print(json.dumps({
    'prompt': 'a calico cat on a windowsill. overall_soundscape: quiet room.',
    'width': 704, 'height': 704, 'num_frames': 56, 'steps': 8, 'seed': 424242,
    'fast': False, 'stream': True, 'preview': ${preview_flag},
    'preview_frames': 1, 'preview_max_side': 256,
  }))")
  local start_line
  start_line=$(wc -l <"$LOG" | tr -d ' ')
  local http_code
  http_code=$(curl -s -N --max-time 3600 -o "$OUT/sse_${tag}.txt" -w "%{http_code}" \
    -X POST "http://127.0.0.1:${PORT}/v1/video/generations" \
    -H 'Content-Type: application/json' \
    -d "$body") || true
  echo "${tag} http=${http_code}" >&2
  sleep 1
  tail -n +"$((start_line + 1))" "$LOG" >"$OUT/log_${tag}.txt"
  python3 - "$OUT/sse_${tag}.txt" "$OUT/preview_${tag}.meta.json" <<'PY' || true
import base64, json, re, sys
from pathlib import Path
sse_path, meta_path = sys.argv[1], sys.argv[2]
text = Path(sse_path).read_text(errors="replace")
best = None
for line in text.splitlines():
    if not line.startswith("data: "):
        continue
    payload = line[6:].strip()
    if payload == "[DONE]":
        continue
    try:
        ev = json.loads(payload)
    except json.JSONDecodeError:
        continue
    if ev.get("type") != "progress" or "preview" not in ev:
        continue
    best = ev
if not best:
    Path(meta_path).write_text(json.dumps({"error": "no preview progress event"}))
    sys.exit(0)
raw = base64.standard_b64decode(best["preview"])
Path(meta_path).write_text(json.dumps({
    "step": best.get("step"),
    "w": best.get("w"),
    "h": best.get("h"),
    "jpeg_bytes": len(raw),
}, indent=2))
PY
}

echo "=== Arm A: preview OFF ===" >&2
start_server
run_job off False

echo "=== Arm B: preview ON, linear (MINIMAX_H3_TAE=0) ===" >&2
start_server MINIMAX_H3_PREVIEW_PROF=1 MINIMAX_H3_TAE=0
run_job linear True

echo "=== Arm C: preview ON, Core ML TAE / ANE ===" >&2
if [[ -n "${MINIMAX_H3_TAE_MODEL:-}" ]]; then
  start_server MINIMAX_H3_PREVIEW_PROF=1 MINIMAX_H3_TAE_MODEL="$TAE_MODEL"
else
  start_server MINIMAX_H3_PREVIEW_PROF=1
fi
run_job tae_ane True

python3 - <<'PY' "$OUT" "$TAE_MODEL"
import json, re, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
tae_model = sys.argv[2]

def parse_steps(path: Path):
    text = path.read_text()
    by_step = {}
    for m in re.finditer(r"\[minimax-h3\] step (\d+)/(\d+) sigma [^\n]+ \((\d+) ms\)", text):
        by_step[int(m.group(1))] = int(m.group(3))
    steps = [by_step[k] for k in sorted(by_step)]
    cached = len(re.findall(r"cached velocity", text))
    samp = re.search(r"\[minimax-h3\] sampling done in (\d+) ms", text)
    resident = re.search(r"\[minimax-h3\] dit resident: ([0-9.]+) GB", text)
    gen400 = re.search(r"\[gen\] 400: ([^\n]+)", text)
    return {
        "step_ms": steps,
        "n_steps": len(steps),
        "cached_velocity_steps": cached,
        "sampling_total_ms": int(samp.group(1)) if samp else None,
        "dit_resident_gb": float(resident.group(1)) if resident else None,
        "gen_400": gen400.group(1) if gen400 else None,
    }

def parse_preview_prof(path: Path):
    text = path.read_text()
    tae_lines = [ln for ln in text.splitlines() if "path=tae" in ln]
    linear = [ln for ln in text.splitlines() if "[h3-preview-prof]" in ln and "path=tae" not in ln]
    h3_tae_warn = len(re.findall(r"\[h3-tae\]", text))
    def bucket(line, key):
        m = re.search(rf"{key}=(\d+)", line)
        return int(m.group(1)) if m else None
    prof = {"linear_lines": len(linear), "tae_lines": len(tae_lines), "h3_tae_warn": h3_tae_warn}
    if linear:
        ln = linear[-1]
        prof["linear_sample"] = {
            "decode_ms": bucket(ln, "decode_ms"),
            "jpeg_ms": bucket(ln, "jpeg_ms"),
            "sync_ms": bucket(ln, "sync_ms"),
            "rollup": bucket(ln, "rollup unpatchify\\+slice") or bucket(ln, "unpatchify_build_ms"),
        }
    if tae_lines:
        ln = tae_lines[-1]
        prof["tae_sample"] = {
            "coreml_ms": bucket(ln, "coreml_ms"),
            "jpeg_ms": bucket(ln, "jpeg_ms"),
            "vol_copy_ms": bucket(ln, "vol_copy_ms"),
        }
    return prof

def mean(xs):
    return statistics.mean(xs) if xs else None

arms = {}
for tag in ("off", "linear", "tae_ane"):
    p = out / f"log_{tag}.txt"
    if not p.exists():
        arms[tag] = {"error": "missing log"}
        continue
    d = parse_steps(p)
    d["step_ms_mean"] = mean(d["step_ms"])
    meta_p = out / f"preview_{tag}.meta.json"
    if meta_p.exists():
        d["preview_meta"] = json.loads(meta_p.read_text())
    if tag in ("linear", "tae_ane"):
        d["preview_prof"] = parse_preview_prof(p)
    arms[tag] = d

m_off = arms.get("off", {}).get("step_ms_mean")
m_lin = arms.get("linear", {}).get("step_ms_mean")
m_tae = arms.get("tae_ane", {}).get("step_ms_mean")

def delta_pct(base, other):
    if base and other and base > 0:
        return 100.0 * (other - base) / base
    return None

report = {
    "status": "complete",
    "canvas": "704x704 x 56f, 8 steps, fast:false, stream:true, seed 424242",
    "tae_model": tae_model,
    "arms": arms,
    "vs_preview_off": {
        "linear_step_ms_delta": None if m_off is None or m_lin is None else m_lin - m_off,
        "linear_step_ms_delta_pct": delta_pct(m_off, m_lin),
        "tae_step_ms_delta": None if m_off is None or m_tae is None else m_tae - m_off,
        "tae_step_ms_delta_pct": delta_pct(m_off, m_tae),
    },
    "linear_vs_tae": {
        "step_ms_delta": None if m_lin is None or m_tae is None else m_tae - m_lin,
        "step_ms_delta_pct": delta_pct(m_lin, m_tae),
    },
    "note": "step ms = full denoise iteration (DiT + preview + Euler + eval). Deduped by step index.",
}

(out / "report_tae_ab.json").write_text(json.dumps(report, indent=2))

md = []
md.append("# H3 preview three-arm — linear vs ANE TAE\n")
md.append(f"**Model:** `{tae_model}`  \n**Canvas:** {report['canvas']}\n")
md.append("| Arm | Mean step (ms) | Sampling total (ms) | Preview JPEG (last progress) |")
md.append("|-----|----------------|---------------------|------------------------------|")
for label, tag in [("A preview off", "off"), ("B linear (`TAE=0`)", "linear"), ("C TAE ANE", "tae_ane")]:
    a = arms.get(tag, {})
    pm = (a.get("preview_meta") or {})
    pj = "—" if tag == "off" else f"{pm.get('w','?')}×{pm.get('h','?')} ({pm.get('jpeg_bytes','?')} B)"
    m = a.get("step_ms_mean")
    ms = f"{m:.0f}" if m is not None else "?"
    md.append(f"| {label} | {ms} | {a.get('sampling_total_ms','?')} | {pj} |")

v = report["vs_preview_off"]
md.append("\n## vs preview off\n")
md.append(f"- Linear: **{v['linear_step_ms_delta']:+.0f} ms/step** ({v['linear_step_ms_delta_pct']:+.1f}%)"
          if v["linear_step_ms_delta"] is not None else "- Linear: n/a")
md.append(f"- TAE ANE: **{v['tae_step_ms_delta']:+.0f} ms/step** ({v['tae_step_ms_delta_pct']:+.1f}%)"
          if v["tae_step_ms_delta"] is not None else "- TAE ANE: n/a")
lv = report["linear_vs_tae"]
if lv["step_ms_delta"] is not None:
    md.append(f"\n## Linear vs TAE ANE\n\n**{lv['step_ms_delta']:+.0f} ms/step** ({lv['step_ms_delta_pct']:+.1f}%)")

for tag, title in [("linear", "B linear prof sample"), ("tae_ane", "C TAE prof sample")]:
    prof = arms.get(tag, {}).get("preview_prof")
    if prof:
        md.append(f"\n### {title}\n\n```json\n{json.dumps(prof, indent=2)}\n```")

(out / "TAE_AB_REPORT.md").write_text("\n".join(md) + "\n")
print(json.dumps(report, indent=2))
print(f"\nWrote {out / 'report_tae_ab.json'} and {out / 'TAE_AB_REPORT.md'}", file=sys.stderr)
PY

stop_server
trap - EXIT

HYBRID="${H3_TOOLS_ROOT:-$REPO/../h3-tools}/start-minimax-h3-hybrid.sh"
if [[ "${RESTART_HYBRID_AFTER:-1}" != "0" && -x "$HYBRID" ]]; then
  echo "=== Restarting hybrid mlx-serve on 11361 ===" >&2
  pkill -f 'mlx-serve.*--port 11361' 2>/dev/null || true
  sleep 2
  nohup "$HYBRID" >>"$HOME/.mlx-serve/logs/hybrid-11361-restart.log" 2>&1 &
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:11361/health" >/dev/null 2>&1; then
      echo "hybrid health ok" >&2
      break
    fi
    sleep 2
  done
fi
