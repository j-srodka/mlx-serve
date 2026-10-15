#!/usr/bin/env python3
"""MiniMax H3 project timeline: take, extend (latent pin), export.

Uses mlx-serve /v1/video/generations with latent_out and latent_prefix_path.
One POST per generation — no ComfyUI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

# Shared trim math with src/minimax_h3.zig (FRAME_PER_TOKEN + 17k+5 floor).
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
DEFAULT_OVERLAP = 22


def align_ref_frame_count(n: int) -> int:
    if n < 5:
        return 0
    v = n
    while v % 17 != 5:
        v -= 1
    return v


def video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((frame_count - 5) // 17) * 5 + 2


def snap_keep_end_frame(requested: int, clip_frames: int) -> int:
    cum = 0
    t = 0
    best = 0
    while cum <= clip_frames:
        span = FRAME_PER_TOKEN[t % len(FRAME_PER_TOKEN)]
        cum += span
        if cum <= requested:
            best = cum
        else:
            break
        t += 1
    return best


def keep_end_trim_error(snapped: int, overlap: int = DEFAULT_OVERLAP) -> str | None:
    if snapped < overlap:
        return (
            f"trim snaps to {snapped} source frames, below overlap {overlap} — "
            "need at least 22 frames kept after trim"
        )
    return None


def video_latent_hi_exclusive_for_keep_end(keep_end: int) -> int:
    if keep_end == 0:
        return 0
    if keep_end == 1:
        return 1
    floor = align_ref_frame_count(max(5, keep_end))
    if floor == 0:
        return 0
    base = video_latent_t(floor)
    return base + 1 if keep_end > floor else base


def temporal_shape_audio_t(frame_count: int) -> int:
    if frame_count == 1:
        return round(1 / 24 * 40)
    aligned = frame_count
    while aligned % 17 != 5:
        aligned += 1
    return round(aligned / 24 * 40)


def audio_latent_hi_exclusive_for_keep_end(keep_end: int) -> int:
    if keep_end == 0:
        return 0
    if keep_end == 1:
        return temporal_shape_audio_t(1)
    floor = align_ref_frame_count(max(5, keep_end))
    if floor == 0:
        return 0
    return temporal_shape_audio_t(floor)


def overlap_video_latent_t(overlap_src: int) -> int:
    return video_latent_t(overlap_src)


def overlap_audio_latent_t(overlap_src: int) -> int:
    return temporal_shape_audio_t(overlap_src)


def trim_latent_sidecar(
    video: dict,
    audio: dict,
    keep_end: int,
) -> tuple[dict, dict]:
    """Slice safetensors tensors to trimmed T (numpy arrays in header dict)."""
    import numpy as np

    v = np.array(video["data"]).reshape(video["shape"]).astype("float32")
    a = np.array(audio["data"]).reshape(audio["shape"]).astype("float32")
    v_hi = video_latent_hi_exclusive_for_keep_end(keep_end)
    a_hi = audio_latent_hi_exclusive_for_keep_end(keep_end)
    v_trim = v[:, :, :v_hi, ...].copy()
    a_trim = a[:, :, :, :a_hi].copy()
    return (
        {"shape": list(v_trim.shape), "data": v_trim.flatten().tolist()},
        {"shape": list(a_trim.shape), "data": a_trim.flatten().tolist()},
    )


def load_sidecar_tensors(path: Path):
    import numpy as np
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata() or {}
        if meta.get("pack") != "fl2va":
            raise SystemExit(f"{path}: sidecar must declare pack fl2va")
        if "video_latent" not in f.keys() or "audio_latent" not in f.keys():
            raise SystemExit(f"{path}: sidecar missing video_latent or audio_latent")
        v = np.array(f.get_tensor("video_latent"))
        a = np.array(f.get_tensor("audio_latent"))
    return v, a


def write_sidecar(path: Path, video, audio) -> None:
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"video_latent": video, "audio_latent": audio},
        str(path),
        metadata={"pack": "fl2va"},
    )


@dataclass
class ProjectState:
    root: Path
    slots: list[str]
    canonical: dict[str, str]

    @classmethod
    def load(cls, root: Path) -> ProjectState:
        pj = root / "project.json"
        if not pj.exists():
            return cls(root=root, slots=[], canonical={})
        data = json.loads(pj.read_text())
        return cls(
            root=root,
            slots=list(data.get("slots", [])),
            canonical=dict(data.get("canonical", {})),
        )

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "project.json").write_text(
            json.dumps({"slots": self.slots, "canonical": self.canonical}, indent=2) + "\n"
        )


def new_take_id() -> str:
    return uuid.uuid4().hex[:12]


def cmd_take(args: argparse.Namespace) -> None:
    from generate_h3 import request_video, response_to_mp4

    project = ProjectState.load(args.project)
    slot = args.slot or (project.slots[-1] if project.slots else "slot0")
    if slot not in project.slots:
        project.slots.append(slot)

    take_id = new_take_id()
    take_dir = args.project / "takes" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    latent_path = take_dir / "latents.safetensors"

    body_extra = {
        "latent_out": str(latent_path.resolve()),
        "chain_windows": 1,
    }
    data = _request_with_extras(args, body_extra)
    response_to_mp4(data, take_dir / "clip.mp4")

    keep_end = int(data["frames"])
    take_meta = {
        "id": take_id,
        "slot": slot,
        "prompt": args.prompt,
        "seed": args.seed,
        "parent_id": None,
        "keep_end_frame": keep_end,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "overlap": 0,
    }
    (take_dir / "take.json").write_text(json.dumps(take_meta, indent=2) + "\n")
    project.canonical[slot] = take_id
    project.save()
    print(f"take {take_id} -> {take_dir} (canonical for {slot})")


def cmd_extend(args: argparse.Namespace) -> None:
    from generate_h3 import request_video, response_to_mp4

    parent_dir = args.project / "takes" / args.from_take
    if not parent_dir.is_dir():
        raise SystemExit(f"unknown take: {args.from_take}")
    parent_meta = json.loads((parent_dir / "take.json").read_text())
    clip_frames = int(parent_meta["keep_end_frame"])
    snapped = snap_keep_end_frame(args.trim_end, clip_frames)
    if err := keep_end_trim_error(snapped, args.overlap):
        raise SystemExit(err)
    if snapped != args.trim_end:
        print(f"trim snapped {args.trim_end} -> {snapped} (temporal-token boundary)", file=sys.stderr)

    sidecar = parent_dir / "latents.safetensors"
    if not sidecar.is_file():
        raise SystemExit(f"missing {sidecar} — only script-generated takes can extend")

    v, a = load_sidecar_tensors(sidecar)
    v_hi = video_latent_hi_exclusive_for_keep_end(snapped)
    a_hi = audio_latent_hi_exclusive_for_keep_end(snapped)
    v_trim = v[:, :, :v_hi, ...].copy()
    a_trim = a[:, :, :, :a_hi].copy()

    take_id = new_take_id()
    take_dir = args.project / "takes" / take_id
    take_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = take_dir / "prefix_latents.safetensors"
    write_sidecar(prefix_path, v_trim, a_trim)

    out_latent = take_dir / "latents.safetensors"
    body_extra = {
        "latent_prefix_path": str(prefix_path.resolve()),
        "latent_prefix_frames": args.overlap,
        "latent_out": str(out_latent.resolve()),
        "chain_windows": 1,
    }
    data = _request_with_extras(args, body_extra)
    response_to_mp4(data, take_dir / "clip.mp4")

    slot = parent_meta.get("slot", "slot0")
    project = ProjectState.load(args.project)
    if slot not in project.slots:
        project.slots.append(slot)
    take_meta = {
        "id": take_id,
        "slot": slot,
        "prompt": args.prompt,
        "seed": args.seed,
        "parent_id": args.from_take,
        "keep_end_frame": int(data["frames"]),
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "overlap": args.overlap,
        "trim_from_parent": snapped,
    }
    (take_dir / "take.json").write_text(json.dumps(take_meta, indent=2) + "\n")
    project.canonical[slot] = take_id
    project.save()
    print(f"extend {take_id} from {args.from_take} (parent files kept)")


def export_plan(project: Path) -> list[dict]:
    """Per-slot concat plan: trim ranges and overlap drops (frame indices)."""
    state = ProjectState.load(project)
    plan: list[dict] = []
    for slot in state.slots:
        take_id = state.canonical.get(slot)
        if not take_id:
            continue
        chain: list[dict] = []
        cur = take_id
        while cur:
            td = project / "takes" / cur
            meta = json.loads((td / "take.json").read_text())
            chain.append(meta)
            cur = meta.get("parent_id")
        chain.reverse()
        for i, meta in enumerate(chain):
            entry = {
                "slot": slot,
                "take_id": meta["id"],
                "clip": str((project / "takes" / meta["id"] / "clip.mp4").resolve()),
                "trim_start": 0,
                "trim_end": meta["keep_end_frame"],
            }
            if i > 0:
                ov = int(meta.get("overlap", DEFAULT_OVERLAP))
                entry["trim_start"] = ov
            plan.append(entry)
    return plan


def cmd_export(args: argparse.Namespace) -> None:
    plan = export_plan(args.project)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found — install ffmpeg to export, or use --dry-run")
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="h3-timeline-export-") as tmp:
        tmpdir = Path(tmp)
        parts: list[Path] = []
        for i, seg in enumerate(plan):
            part = tmpdir / f"seg_{i:03d}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    seg["clip"],
                    "-vf",
                    f"trim=start_frame={seg['trim_start']}:end_frame={seg['trim_end']},setpts=PTS-STARTPTS",
                    "-an",
                    str(part),
                ],
                check=True,
            )
            parts.append(part)
        list_file = tmpdir / "concat.txt"
        list_file.write_text("\n".join(f"file '{p}'" for p in parts) + "\n")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                str(out),
            ],
            check=True,
        )
    print(f"exported {out}")


def _request_with_extras(args: argparse.Namespace, extras: dict) -> dict:
    if getattr(args, "dry_run", False):
        body = {
            "prompt": args.prompt,
            "width": args.width,
            "height": args.height,
            "num_frames": args.frames,
            "steps": args.steps,
            "fast": args.fast,
            **extras,
        }
        if args.seed is not None:
            body["seed"] = args.seed
        print(json.dumps(body))
        return {"frames": args.frames, "width": args.width, "height": args.height, "format": "rgb8", "data": ""}

    from generate_h3 import request_video

    return request_video(
        base_url=args.url,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        steps=args.steps,
        seed=args.seed,
        fast=args.fast,
        timeout=7200,
        turbo=args.turbo,
        chain_windows=int(extras.get("chain_windows", 1)),
        latent_prefix_frames=int(extras.get("latent_prefix_frames", 0)),
        latent_out=extras.get("latent_out"),
        latent_prefix_path=extras.get("latent_prefix_path"),
    )


def add_gen_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("-p", "--prompt", required=True)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--frames", type=int, default=56)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--turbo", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print JSON body only; one POST recorded by tests")


def main() -> None:
    ap = argparse.ArgumentParser(description="H3 timeline: take / extend / export")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("take", help="One generation + latent sidecar")
    pt.add_argument("project", type=Path)
    pt.add_argument("--slot", default=None)
    add_gen_flags(pt)
    pt.set_defaults(func=cmd_take)

    pe = sub.add_parser("extend", help="Pin-extend from a prior take")
    pe.add_argument("project", type=Path)
    pe.add_argument("--from", dest="from_take", required=True)
    pe.add_argument("--trim-end", type=int, required=True)
    pe.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    add_gen_flags(pe)
    pe.set_defaults(func=cmd_extend)

    px = sub.add_parser("export", help="Concat canonical clips")
    px.add_argument("project", type=Path)
    px.add_argument("-o", "--output", type=Path, required=True)
    px.add_argument("--dry-run", action="store_true")
    px.set_defaults(func=cmd_export)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
