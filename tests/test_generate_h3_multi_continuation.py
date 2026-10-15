#!/usr/bin/env python3
"""Hermetic tests for generate_h3_multi continuation wiring (no mlx-serve)."""

from __future__ import annotations

import subprocess
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_h3_multi import (  # noqa: E402
    CONTINUITY_PROMPT_ONLY,
    CONTINUITY_VIDEO,
    ClipJob,
    Manifest,
    build_ref_videos_for_clip,
    load_manifest,
    parse_frontmatter,
    resolve_clip_continuity,
    resolve_continuation_tail_seconds,
)


class ContinuityParsingTests(unittest.TestCase):
    def test_mixed_manifest_frontmatter(self) -> None:
        text = """---
continuation_tail_seconds: 2
clips:
  - file: 01.txt
    duration: 10
  - file: 02.txt
    duration: 8
    continuity: video_continuation
  - file: 03.txt
    duration: 6
    continuity: prompt_only
---
# body
"""
        data = parse_frontmatter(text)
        self.assertEqual(data["continuation_tail_seconds"], 2)
        self.assertEqual(data["clips"][1]["continuity"], "video_continuation")
        self.assertEqual(data["clips"][2]["continuity"], "prompt_only")

    def test_resolve_tail_default(self) -> None:
        self.assertEqual(resolve_continuation_tail_seconds(None, None), 2.0)

    def test_resolve_tail_from_manifest(self) -> None:
        m = Manifest(
            raw={},
            clips=[],
            width=None,
            height=None,
            ref_max_edge=None,
            continuity_default=None,
            continuation_tail_seconds=3.0,
        )
        self.assertEqual(resolve_continuation_tail_seconds(m, None), 3.0)

    def test_resolve_clip_continuity_mixed(self) -> None:
        manifest = Manifest(
            raw={},
            clips=[
                {"file": "01.txt", "duration": 10},
                {"file": "02.txt", "duration": 8, "continuity": "video_continuation"},
                {"file": "03.txt", "duration": 6, "continuity": "prompt_only"},
            ],
            width=None,
            height=None,
            ref_max_edge=None,
            continuity_default=None,
            continuation_tail_seconds=2.0,
        )
        self.assertIsNone(
            resolve_clip_continuity(0, manifest.clips[0], manifest, None)
        )
        self.assertEqual(
            resolve_clip_continuity(1, manifest.clips[1], manifest, None),
            CONTINUITY_VIDEO,
        )
        self.assertEqual(
            resolve_clip_continuity(2, manifest.clips[2], manifest, None),
            CONTINUITY_PROMPT_ONLY,
        )

    def test_resolve_clip_continuity_default_video(self) -> None:
        self.assertEqual(
            resolve_clip_continuity(1, {}, None, None),
            CONTINUITY_VIDEO,
        )


class ContinuationRefWiringTests(unittest.TestCase):
    def test_prompt_only_skips_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            prior = run_dir / "01.mp4"
            prior.write_bytes(b"not-a-real-mp4")
            job = ClipJob(
                path=run_dir / "02.txt",
                duration_s=8.0,
                frames=120,
                continuity=CONTINUITY_PROMPT_ONLY,
            )
            refs, cont_ref = build_ref_videos_for_clip(
                clip_index=1,
                job=job,
                prior_output=prior,
                tail_seconds=2.0,
                run_dir=run_dir,
                static_ref_videos=[],
            )
            self.assertEqual(refs, [])
            self.assertIsNone(cont_ref)

    def test_video_continuation_extracts_tail(self) -> None:
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not available")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            prior = run_dir / "01.mp4"
            # 3s black test clip
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(prior),
                ],
                check=True,
            )
            job = ClipJob(
                path=run_dir / "02.txt",
                duration_s=8.0,
                frames=120,
                continuity=CONTINUITY_VIDEO,
            )
            refs, cont_ref = build_ref_videos_for_clip(
                clip_index=1,
                job=job,
                prior_output=prior,
                tail_seconds=2.0,
                run_dir=run_dir,
                static_ref_videos=[],
            )
            self.assertEqual(len(refs), 1)
            self.assertIsNotNone(cont_ref)
            self.assertTrue(cont_ref.is_file())
            self.assertGreater(cont_ref.stat().st_size, 0)

    def test_load_manifest_from_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clips_dir = Path(tmp)
            (clips_dir / "manifest.md").write_text(
                """---
continuation_tail_seconds: 2
clips:
  - file: 01.txt
    duration: 10
  - file: 02.txt
    duration: 8
    continuity: video_continuation
---
# test
""",
                encoding="utf-8",
            )
            (clips_dir / "01.txt").write_text("prompt1", encoding="utf-8")
            (clips_dir / "02.txt").write_text("prompt2", encoding="utf-8")
            m = load_manifest(clips_dir)
            self.assertIsNotNone(m)
            assert m is not None
            self.assertEqual(m.continuation_tail_seconds, 2.0)
            self.assertEqual(m.clips[1]["continuity"], "video_continuation")


class RefModCliTests(unittest.TestCase):
    def test_multi_help_lists_ref_mod(self) -> None:
        out = subprocess.run(
            [sys.executable, str(ROOT / "generate_h3_multi.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(out.returncode, 0)
        self.assertIn("--ref-mod", out.stdout)


if __name__ == "__main__":
    unittest.main()
