#!/usr/bin/env python3
"""Smoke test for minimax-h3-stills crop_sheet.py (ffmpeg only)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

CROP = Path.home() / ".agents/skills/minimax-h3-stills/scripts/crop_sheet.py"
OUT = Path("/tmp/test_sheet.png")
CROPS = Path("/tmp/test_crops")


def main() -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=1344x768:d=1",
            "-frames:v",
            "1",
            str(OUT),
        ],
        check=True,
    )

    spec = importlib.util.spec_from_file_location("crop_sheet", CROP)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    paths = mod.crop_panels(OUT, CROPS)
    assert len(paths) == 8
    assert paths[0].name == "front.png"
    assert paths[-1].name == "leg.png"
    print("crop_sheet ok:", len(paths), "panels")


if __name__ == "__main__":
    main()
    sys.exit(0)
