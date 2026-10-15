#!/usr/bin/env python3
"""Hermetic tests for generate_h3 chain / latent-prefix CLI (no mlx-serve)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_h3 import DEFAULT_AB_LATENT_PREFIX_FRAMES, request_video, snap_frames  # noqa: E402


class RequestVideoBodyTests(unittest.TestCase):
    def test_chain_and_latent_prefix_fields(self) -> None:
        captured: dict = {}

        def fake_urlopen(req, timeout=0):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode())
            resp = MagicMock()
            resp.read.return_value = json.dumps(
                {
                    "format": "rgb8",
                    "frames": 1,
                    "width": 32,
                    "height": 32,
                    "data": "AA==",
                }
            ).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", fake_urlopen):
            request_video(
                base_url="http://127.0.0.1:1",
                prompt="x",
                width=32,
                height=32,
                num_frames=56,
                steps=10,
                seed=42,
                fast=False,
                timeout=5,
                chain_windows=2,
                latent_prefix_frames=22,
            )

        body = captured["body"]
        self.assertEqual(body["chain_windows"], 2)
        self.assertEqual(body["latent_prefix_frames"], 22)
        self.assertNotIn("turbo", body)

    def test_omits_chain_when_single_window(self) -> None:
        captured: dict = {}

        def fake_urlopen(req, timeout=0):  # noqa: ARG001
            captured["body"] = json.loads(req.data.decode())
            resp = MagicMock()
            resp.read.return_value = json.dumps(
                {
                    "format": "rgb8",
                    "frames": 1,
                    "width": 32,
                    "height": 32,
                    "data": "AA==",
                }
            ).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", fake_urlopen):
            request_video(
                base_url="http://127.0.0.1:1",
                prompt="x",
                width=32,
                height=32,
                num_frames=56,
                steps=10,
                seed=None,
                fast=True,
                timeout=5,
                chain_windows=1,
                latent_prefix_frames=0,
            )

        body = captured["body"]
        self.assertNotIn("chain_windows", body)
        self.assertNotIn("latent_prefix_frames", body)


class ArgparseTests(unittest.TestCase):
    def run_h3(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "generate_h3.py"), *extra],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_help_lists_chain_flags(self) -> None:
        out = self.run_h3("--help")
        self.assertEqual(out.returncode, 0)
        self.assertIn("--chain-windows", out.stdout)
        self.assertIn("--latent-prefix-frames", out.stdout)
        self.assertIn("--ab", out.stdout)

    def test_latent_prefix_requires_chain(self) -> None:
        out = self.run_h3(
            "-p",
            "x",
            "--latent-prefix-frames",
            "22",
        )
        self.assertIn(
            "--latent-prefix-frames requires --chain-windows > 1",
            out.stderr + out.stdout,
        )

    def test_ab_requires_output_dir(self) -> None:
        out = self.run_h3(
            "-p",
            "x",
            "--chain-windows",
            "2",
            "--ab",
        )
        self.assertIn("--ab requires -o OUTPUT_DIR", out.stderr + out.stdout)

    def test_snap_frames_ab_default_prefix_constant(self) -> None:
        self.assertEqual(snap_frames(56), 56)
        self.assertEqual(DEFAULT_AB_LATENT_PREFIX_FRAMES, 22)


if __name__ == "__main__":
    unittest.main()
