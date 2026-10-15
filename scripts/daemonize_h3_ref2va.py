#!/usr/bin/env python3
"""Daemonize mlx-serve H3 Ref2VA so lean-ctx 120s cannot kill it."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path.home() / "Github/mlx-serve"
BIN = REPO / "zig-out/bin/mlx-serve"
MODEL = Path.home() / ".mlx-serve/models/ddalcu/MiniMax-H3-REF2VA-MLX-Serve-8bit"
LOG = Path.home() / ".mlx-serve/logs/mlx-serve-11361-stills.log"
PIDFILE = Path.home() / ".mlx-serve/logs/mlx-serve-11361-stills.pid"

LOG.parent.mkdir(parents=True, exist_ok=True)

args = [
    str(BIN),
    "--model",
    str(MODEL),
    "--serve",
    "--host",
    "0.0.0.0",
    "--port",
    "11361",
    "--skip-mem-preflight",
    "--timeout",
    "3600",
]


def daemonize() -> None:
    if os.fork() > 0:
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    os.chdir(str(REPO))
    os.environ["MLX_SERVE_BIN"] = str(BIN)
    log = open(LOG, "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdin.close()
    PIDFILE.write_text(str(os.getpid()))
    os.execv(args[0], args)


if __name__ == "__main__":
    if not BIN.is_file():
        raise SystemExit(f"missing {BIN}")
    daemonize()
    print(f"started pidfile={PIDFILE} log={LOG}")
