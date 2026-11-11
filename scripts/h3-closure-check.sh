#!/usr/bin/env bash
# Fail if H3 / ornith load-bearing paths disappear (post-merge / upstream refresh sanity).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TOOLS="${H3_TOOLS_ROOT:-$(cd "$ROOT/.." && pwd)/h3-tools}"

missing=0
need() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "h3-closure-check: missing: $p" >&2
    missing=1
  fi
}

need_tool() {
  local rel="$1"
  if [[ -e "$ROOT/$rel" ]]; then
    return 0
  fi
  if [[ -e "$TOOLS/$rel" ]]; then
    return 0
  fi
  echo "h3-closure-check: missing: $rel (mlx-serve or $TOOLS)" >&2
  missing=1
}

# src: h3 entrypoints
while IFS= read -r -d '' f; do
  need "${f#"$ROOT"/}"
done < <(find "$ROOT/src" -maxdepth 1 -name 'h3*.zig' -print0 2>/dev/null || true)

need src/h3_refmod.zig
need src/h3_tae.zig
need src/plugins/minimax_h3/h3.zig
need src/plugins/minimax_h3/plugin.zig

# CoreML TAE bridge
need lib/h3_tae/h3_tae_preview.m
need lib/h3_tae/h3_tae_preview.h

# Build wiring
if ! grep -q addH3TaeSources "$ROOT/build/libs.zig" "$ROOT/build/macos.zig" 2>/dev/null; then
  echo "h3-closure-check: addH3TaeSources not referenced in build/libs.zig + build/macos.zig" >&2
  missing=1
fi

# Launchers and timeline tooling (live in ../h3-tools after hygiene split)
need_tool start-minimax-h3.sh
need start-minimax-h3-hybrid.sh
need_tool start-minimax-h3-ref2va.sh
need_tool h3_timeline.py
need_tool generate_h3.py
need_tool generate_h3_multi.py

# Tests
while IFS= read -r -d '' f; do
  need "${f#"$ROOT"/}"
done < <(find "$ROOT/tests" -maxdepth 1 -name 'test_h3_*' -print0 2>/dev/null || true)

while IFS= read -r -d '' f; do
  need_tool "${f#"$ROOT"/}"
done < <(find "$ROOT/tests" -maxdepth 1 -name 'test_generate_h3*.py' -print0 2>/dev/null || true)

# Topology docs and push guard (untracked locally; copied to h3-tools/hygiene/)
for p in .cursor/rules/00-repo-topology.mdc docs/gotchas/repo-topology.md .githooks/pre-push; do
  need "$p"
done

if [[ "$missing" -ne 0 ]]; then
  echo "h3-closure-check: failed (${missing} problem(s))" >&2
  exit 1
fi

echo "h3-closure-check: OK"
