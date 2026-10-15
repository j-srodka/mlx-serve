"""Hermetic tests for h3_timeline.py (trim math, project pointer, export plan)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h3_timeline as ht  # noqa: E402


class TrimMathTests(unittest.TestCase):
    def test_snap_fixtures_141_frame_clip(self) -> None:
        clip = 141
        self.assertEqual(ht.snap_keep_end_frame(100, clip), 98)
        self.assertEqual(ht.snap_keep_end_frame(31, clip), 30)
        self.assertEqual(ht.snap_keep_end_frame(22, clip), 22)
        self.assertEqual(ht.snap_keep_end_frame(141, clip), 141)
        self.assertIsNotNone(ht.keep_end_trim_error(ht.snap_keep_end_frame(21, clip)))

    def test_pin_indices_trim_98(self) -> None:
        keep = 98
        overlap = 22
        pin_v = ht.overlap_video_latent_t(overlap)
        v_hi = ht.video_latent_hi_exclusive_for_keep_end(keep)
        self.assertEqual(pin_v, 7)
        self.assertEqual(v_hi, 28)
        self.assertEqual(v_hi - pin_v, 21)
        self.assertEqual(v_hi - 1, 27)


class ProjectPointerTests(unittest.TestCase):
    def test_canonical_moves_on_new_take(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            state = ht.ProjectState.load(project)
            state.slots = ["a"]
            state.canonical = {"a": "take_old"}
            state.save()
            take_old = project / "takes" / "take_old"
            take_old.mkdir(parents=True)
            (take_old / "take.json").write_text(
                json.dumps({"id": "take_old", "keep_end_frame": 56, "slot": "a"})
            )
            (take_old / "latents.safetensors").write_bytes(b"placeholder")

            take_new = project / "takes" / "take_new"
            take_new.mkdir(parents=True)
            (take_new / "take.json").write_text(
                json.dumps(
                    {
                        "id": "take_new",
                        "parent_id": "take_old",
                        "keep_end_frame": 56,
                        "slot": "a",
                        "overlap": 22,
                    }
                )
            )

            state.canonical["a"] = "take_new"
            state.save()

            reloaded = ht.ProjectState.load(project)
            self.assertEqual(reloaded.canonical["a"], "take_new")
            self.assertTrue((take_old / "latents.safetensors").exists())


class ExtendDryRunTests(unittest.TestCase):
    @mock.patch("generate_h3.response_to_mp4")
    @mock.patch("h3_timeline._request_with_extras")
    @mock.patch("h3_timeline.write_sidecar")
    @mock.patch("h3_timeline.load_sidecar_tensors")
    def test_extend_dry_run_one_post_shape(
        self,
        mock_load: mock.MagicMock,
        mock_write: mock.MagicMock,
        mock_req: mock.MagicMock,
        _mock_mp4: mock.MagicMock,
    ) -> None:
        import numpy as np

        mock_load.return_value = (
            np.zeros((1, 24, 42, 2, 2), dtype=np.float32),
            np.zeros((1, 8, 8, 150), dtype=np.float32),
        )
        posted: list[dict] = []

        def capture(_args, extras):
            posted.append(extras)
            return {"frames": 56, "width": 256, "height": 256, "format": "rgb8", "data": ""}

        mock_req.side_effect = capture

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            parent = project / "takes" / "parent1"
            parent.mkdir(parents=True)
            (parent / "take.json").write_text(
                json.dumps({"id": "parent1", "keep_end_frame": 141, "slot": "s0"})
            )
            (parent / "latents.safetensors").write_bytes(b"stub")

            args = argparse.Namespace(
                project=project,
                from_take="parent1",
                trim_end=100,
                overlap=22,
                url="http://127.0.0.1:8080",
                prompt="continue",
                width=256,
                height=256,
                frames=56,
                steps=4,
                seed=1,
                fast=True,
                turbo=False,
                dry_run=True,
            )

            ht.cmd_extend(args)
            self.assertEqual(len(posted), 1)
            body = posted[0]
            self.assertEqual(body.get("chain_windows"), 1)
            self.assertTrue(Path(body["latent_prefix_path"]).is_absolute())
            self.assertEqual(body.get("latent_prefix_frames"), 22)


class ExportPlanTests(unittest.TestCase):
    def test_export_frame_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            for tid, parent, keep, overlap in (
                ("t1", None, 100, 0),
                ("t2", "t1", 80, 22),
            ):
                td = project / "takes" / tid
                td.mkdir(parents=True)
                (td / "clip.mp4").write_bytes(b"")
                (td / "take.json").write_text(
                    json.dumps(
                        {
                            "id": tid,
                            "parent_id": parent,
                            "keep_end_frame": keep,
                            "slot": "main",
                            "overlap": overlap,
                        }
                    )
                )
            state = ht.ProjectState.load(project)
            state.slots = ["main"]
            state.canonical = {"main": "t2"}
            state.save()

            plan = ht.export_plan(project)
            self.assertEqual(len(plan), 2)
            self.assertEqual(plan[0]["trim_end"], 100)
            self.assertEqual(plan[0]["trim_start"], 0)
            self.assertEqual(plan[1]["trim_end"], 80)
            self.assertEqual(plan[1]["trim_start"], 22)


if __name__ == "__main__":
    unittest.main()
