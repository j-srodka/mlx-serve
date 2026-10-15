#!/usr/bin/env bash
# Delegate to the REF2VA launcher. The FL2VA 4-bit pack
# (~/.mlx-serve/models/ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit) was removed from this
# machine and is not coming back; all H3 serving now goes through the 8-bit REF2VA pack.
# Kept as an entry point so old muscle memory and skill references still work.
exec "$(cd "$(dirname "$0")" && pwd)/start-minimax-h3-ref2va.sh" "$@"
