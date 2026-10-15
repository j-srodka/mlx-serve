#!/usr/bin/env python3
"""Run up to 8 Ref2VA clips with shared settings, optionally concat with ffmpeg.

Requires mlx-serve Ref2VA already running (e.g. ./start-minimax-h3-ref2va.sh).

With a clips directory, optional manifest.md (YAML frontmatter) supplies per-clip
durations (4–15 seconds each) and per-clip continuity for joins. On
video_continuation joins the runner attaches the last N seconds of the prior
clip as --ref-video (default 2s). See minimax-h3-prompts multi-clip workflow.

  generate_h3_multi.py --clips-dir ~/prompts/scene_a \\
    --ref-image face.jpg --ref-image outfit.jpg --concat

  generate_h3_multi.py -p clip01.txt,clip02.txt \\
    --ref-image face.jpg --total-duration 28
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_h3 import (
    IMAGE_EXTS,
    request_video,
    resolve_lora_scales,
    resolve_steps,
    response_to_mp4,
    round32,
    snap_frames,
    validate_lora_paths,
    validate_ref_mod_paths,
)

MAX_CLIPS = 8
MAX_REF_VIDEOS = 3
FPS = 24
MIN_DURATION_S = 4.0
MAX_DURATION_S = 15.0
DEFAULT_CONTINUATION_TAIL_S = 2.0
MIN_CONTINUATION_TAIL_S = 5 / FPS  # ~0.21s, one frame at 24fps
MANIFEST_NAME = "manifest.md"

CONTINUITY_VIDEO = "video_continuation"
CONTINUITY_PROMPT_ONLY = "prompt_only"
VALID_CONTINUITY = {CONTINUITY_VIDEO, CONTINUITY_PROMPT_ONLY}


@dataclass
class ClipJob:
    path: Path
    duration_s: float
    frames: int
    continuity: str | None = None  # clip 2+: video_continuation | prompt_only


@dataclass
class Manifest:
    raw: dict[str, Any]
    clips: list[dict[str, Any]]
    width: int | None
    height: int | None
    ref_max_edge: int | None
    continuity_default: str | None
    continuation_tail_seconds: float | None


def check_server(base_url: str, timeout: int = 10) -> None:
    url = f"{base_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                raise SystemExit(f"Server health check failed: HTTP {resp.status} from {url}")
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Cannot reach mlx-serve at {base_url}\n"
            "Start Ref2VA first: ./start-minimax-h3-ref2va.sh"
        ) from e


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse YAML frontmatter from manifest.md (requires PyYAML or falls back)."""
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    body = parts[1].strip()
    if not body:
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return _parse_frontmatter_minimal(body)
    data = yaml.safe_load(body)
    return data if isinstance(data, dict) else {}


def _parse_frontmatter_minimal(body: str) -> dict[str, Any]:
    """Subset parser for manifest frontmatter when PyYAML is unavailable."""
    result: dict[str, Any] = {}
    clips: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_clips = False
    in_pictures = False
    pictures: list[dict[str, Any]] = []
    pic: dict[str, Any] | None = None

    for line in body.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue

        # Indented fields belong to the current list item.
        if line.startswith("  ") and not line.startswith("  -"):
            key, _, val = line.strip().partition(":")
            if key and val:
                item = _scalar(val.strip())
                if in_clips and current is not None:
                    current[key.strip()] = item
                elif in_pictures and pic is not None:
                    pic[key.strip()] = item
            continue

        stripped = line.strip()

        if stripped == "clips:":
            in_clips = True
            in_pictures = False
            current = None
            continue
        if stripped == "pictures:":
            in_pictures = True
            in_clips = False
            current = None
            continue

        if stripped.startswith("- ") and in_clips:
            if current:
                clips.append(current)
            current = {}
            rest = stripped[2:].strip()
            if rest:
                key, _, val = rest.partition(":")
                if key and val:
                    current[key.strip()] = _scalar(val.strip())
            continue

        if stripped.startswith("- ") and in_pictures:
            if pic:
                pictures.append(pic)
            pic = {}
            rest = stripped[2:].strip()
            if rest:
                key, _, val = rest.partition(":")
                if key and val:
                    pic[key.strip()] = _scalar(val.strip())
            continue

        if re.match(r"^[a-zA-Z_][\w]*:", stripped):
            in_clips = False
            in_pictures = False
            key, _, val = stripped.partition(":")
            val = val.strip()
            if val:
                result[key] = _scalar(val)

    if current:
        clips.append(current)
    if pic:
        pictures.append(pic)
    if clips:
        result["clips"] = clips
    if pictures:
        result["pictures"] = pictures
    return result


def _scalar(val: str) -> Any:
    val = val.strip().strip('"').strip("'")
    if val.isdigit():
        return int(val)
    try:
        return float(val)
    except ValueError:
        return val


def load_manifest(clips_dir: Path) -> Manifest | None:
    path = clips_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    data = parse_frontmatter(path.read_text(encoding="utf-8"))
    if not data:
        raise SystemExit(f"could not parse frontmatter in {path}")
    clips = data.get("clips")
    if clips is not None and not isinstance(clips, list):
        raise SystemExit(f"{MANIFEST_NAME}: 'clips' must be a list")
    return Manifest(
        raw=data,
        clips=clips or [],
        width=data.get("width"),
        height=data.get("height"),
        ref_max_edge=data.get("ref_max_edge"),
        continuity_default=_parse_continuity_value(data.get("continuity")),
        continuation_tail_seconds=_parse_tail_seconds(data.get("continuation_tail_seconds")),
    )


def _parse_continuity_value(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    if s not in VALID_CONTINUITY:
        raise SystemExit(
            f"invalid continuity {s!r}; use {CONTINUITY_VIDEO} or {CONTINUITY_PROMPT_ONLY}"
        )
    return s


def _parse_tail_seconds(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        raise SystemExit(f"invalid continuation_tail_seconds: {val!r}")


def resolve_continuation_tail_seconds(
    manifest: Manifest | None,
    cli_tail: float | None,
) -> float:
    if cli_tail is not None:
        tail = cli_tail
    elif manifest and manifest.continuation_tail_seconds is not None:
        tail = manifest.continuation_tail_seconds
    else:
        tail = DEFAULT_CONTINUATION_TAIL_S
    if tail < MIN_CONTINUATION_TAIL_S:
        raise SystemExit(
            f"continuation_tail_seconds {tail}s too short "
            f"(minimum ~{MIN_CONTINUATION_TAIL_S:g}s / 5 frames @ {FPS}fps)"
        )
    if tail > MAX_DURATION_S:
        raise SystemExit(
            f"continuation_tail_seconds {tail}s exceeds ref video max ({MAX_DURATION_S:g}s)"
        )
    return tail


def resolve_clip_continuity(
    clip_index: int,
    clip_entry: dict[str, Any] | None,
    manifest: Manifest | None,
    cli_continuity: str | None,
) -> str | None:
    """Return continuity mode for clip at clip_index (0-based). Clip 0 → None."""
    if clip_index == 0:
        return None
    if clip_entry and clip_entry.get("continuity") is not None:
        return _parse_continuity_value(clip_entry["continuity"])
    if manifest and manifest.continuity_default is not None:
        return manifest.continuity_default
    if cli_continuity is not None:
        return _parse_continuity_value(cli_continuity)
    return CONTINUITY_VIDEO


def validate_duration(seconds: float, *, context: str) -> None:
    if seconds < MIN_DURATION_S or seconds > MAX_DURATION_S:
        raise SystemExit(
            f"{context}: duration {seconds}s out of range "
            f"({MIN_DURATION_S:g}–{MAX_DURATION_S:g}s)"
        )


def duration_to_frames(duration: float) -> int:
    validate_duration(duration, context="clip")
    return snap_frames(round(duration * FPS))


def equal_split_durations(total: float, clip_count: int) -> list[float]:
    if clip_count < 1:
        raise SystemExit("need at least one clip for --total-duration")
    each = total / clip_count
    if each < MIN_DURATION_S - 1e-9:
        raise SystemExit(
            f"--total-duration {total}s across {clip_count} clips = {each:.2f}s each; "
            f"minimum is {MIN_DURATION_S:g}s per clip"
        )
    if each > MAX_DURATION_S + 1e-9:
        raise SystemExit(
            f"--total-duration {total}s across {clip_count} clips = {each:.2f}s each; "
            f"maximum is {MAX_DURATION_S:g}s per clip — add more clips or lower total"
        )
    return [each] * clip_count


def resolve_clip_jobs(
    *,
    clips_dir: Path | None,
    prompt_list: str | None,
    manifest: Manifest | None,
    cli_duration: float | None,
    cli_total_duration: float | None,
    cli_frames: int | None,
    cli_continuity: str | None = None,
) -> list[ClipJob]:
    if clips_dir is not None and prompt_list is not None:
        raise SystemExit("provide only one of --clips-dir or -p (not both)")

    manifest_has_durations = False
    clip_entries: list[dict[str, Any]] = []

    if manifest and manifest.clips:
        clip_entries = manifest.clips
        manifest_has_durations = all(
            isinstance(c.get("duration"), (int, float)) for c in clip_entries
        )
        if any(c.get("duration") is not None for c in clip_entries) and not manifest_has_durations:
            raise SystemExit(
                f"{MANIFEST_NAME}: every clip must have 'duration' when any clip specifies it"
            )

    if manifest_has_durations:
        if cli_duration is not None or cli_total_duration is not None or cli_frames is not None:
            raise SystemExit(
                f"{MANIFEST_NAME} lists per-clip durations; "
                "do not pass --duration, --total-duration, or --frames"
            )

    if clips_dir is not None:
        base = clips_dir
        if clip_entries:
            files: list[Path] = []
            for entry in clip_entries:
                name = entry.get("file")
                if not isinstance(name, str) or not name.strip():
                    raise SystemExit(f"{MANIFEST_NAME}: each clip needs a 'file' name")
                path = (base / name).resolve()
                if not path.is_file():
                    raise SystemExit(f"clip prompt not found: {path} (from {MANIFEST_NAME})")
                files.append(path)
        else:
            files = sorted(
                p for p in base.glob("*.txt")
                if p.is_file() and not p.name.startswith(".") and p.name != MANIFEST_NAME
            )
    elif prompt_list is not None:
        if clip_entries:
            raise SystemExit(f"-p cannot be used with {MANIFEST_NAME} clip list (use --clips-dir)")
        files = []
        for part in prompt_list.split(","):
            part = part.strip()
            if not part:
                continue
            path = Path(part).expanduser()
            if not path.is_file():
                raise SystemExit(f"clip prompt file not found: {path}")
            files.append(path)
    else:
        raise SystemExit("provide --clips-dir DIR or -p file1.txt,file2.txt,...")

    if not files:
        raise SystemExit("no clip prompt files found (need 1–8 .txt files)")
    if len(files) > MAX_CLIPS:
        raise SystemExit(f"at most {MAX_CLIPS} clips; got {len(files)}")

    if cli_duration is not None and cli_total_duration is not None:
        raise SystemExit("use only one of --duration or --total-duration")

    durations: list[float]
    if manifest_has_durations:
        durations = []
        for i, entry in enumerate(clip_entries):
            d = float(entry["duration"])
            validate_duration(d, context=f"{MANIFEST_NAME} clip {i + 1}")
            durations.append(d)
    elif cli_total_duration is not None:
        durations = equal_split_durations(cli_total_duration, len(files))
    elif cli_frames is not None:
        frames = snap_frames(cli_frames)
        d = frames / FPS
        validate_duration(d, context="--frames")
        durations = [d] * len(files)
    elif cli_duration is not None:
        validate_duration(cli_duration, context="--duration")
        durations = [cli_duration] * len(files)
    else:
        validate_duration(15.0, context="default")
        durations = [15.0] * len(files)

    jobs: list[ClipJob] = []
    for i, (path, dur) in enumerate(zip(files, durations, strict=True)):
        frames = duration_to_frames(dur)
        entry = clip_entries[i] if i < len(clip_entries) else None
        continuity = resolve_clip_continuity(
            i,
            entry,
            manifest,
            cli_continuity,
        )
        jobs.append(
            ClipJob(path=path, duration_s=dur, frames=frames, continuity=continuity)
        )
    return jobs


def prepare_ref_images(
    sources: list[Path],
    out_dir: Path,
    max_edge: int,
) -> list[Path]:
    if not sources:
        return []
    if len(sources) > 9:
        raise SystemExit("at most 9 --ref-image files")

    out_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[Path] = []

    for i, src in enumerate(sources, start=1):
        if src.suffix.lower() not in IMAGE_EXTS:
            raise SystemExit(f"Unsupported image type {src.suffix!r}: {src}")
        if not src.is_file():
            raise SystemExit(f"Image not found: {src}")

        dst = out_dir / f"ref_{i:02d}.jpg"
        if max_edge <= 0:
            if src.suffix.lower() in {".jpg", ".jpeg"}:
                shutil.copy2(src, dst)
            else:
                if not shutil.which("ffmpeg"):
                    raise SystemExit("ffmpeg required to convert reference images to JPEG")
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(src), "-q:v", "3", str(dst),
                    ],
                    check=True,
                )
        else:
            if not shutil.which("ffmpeg"):
                raise SystemExit("ffmpeg required for --ref-max-edge resize")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(src),
                    "-vf", f"scale='min({max_edge},iw)':-2",
                    "-q:v", "3", str(dst),
                ],
                check=True,
            )
        prepared.append(dst)
        print(f"Prepared ref {i}/{len(sources)}: {dst.name}", file=sys.stderr)

    return prepared


def extract_tail_video(
    source_mp4: Path,
    tail_seconds: float,
    out_path: Path,
) -> Path:
    """Extract the last tail_seconds of source_mp4 to out_path."""
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg required for continuation tail extract")
    if not source_mp4.is_file():
        raise SystemExit(f"prior clip not found for continuation: {source_mp4}")

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(source_mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        total_s = float(probe.stdout.strip())
    except ValueError:
        raise SystemExit(f"could not read duration of {source_mp4}") from None

    start_s = max(0.0, total_s - tail_seconds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start_s),
            "-i", str(source_mp4),
            "-c", "copy",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def build_ref_videos_for_clip(
    *,
    clip_index: int,
    job: ClipJob,
    prior_output: Path | None,
    tail_seconds: float,
    run_dir: Path,
    static_ref_videos: list[Path],
) -> tuple[list[Path], Path | None]:
    """Return ref video paths and optional continuation_ref path for run.json."""
    continuation_ref: Path | None = None
    auto_refs: list[Path] = []

    if (
        clip_index > 0
        and job.continuity == CONTINUITY_VIDEO
        and prior_output is not None
    ):
        continuation_ref = run_dir / f"continuation_{clip_index:02d}_tail.mp4"
        extract_tail_video(prior_output, tail_seconds, continuation_ref)
        auto_refs.append(continuation_ref)
        print(
            f"Continuation ref: last {tail_seconds:g}s of {prior_output.name} "
            f"-> {continuation_ref.name}",
            file=sys.stderr,
        )

    combined = auto_refs + list(static_ref_videos)
    if len(combined) > MAX_REF_VIDEOS:
        raise SystemExit(
            f"at most {MAX_REF_VIDEOS} --ref-video files per clip; "
            f"got {len(combined)} ({len(auto_refs)} auto continuation + "
            f"{len(static_ref_videos)} static)"
        )
    return combined, continuation_ref


def resolve_seeds(
    *,
    clip_count: int,
    seed: int | None,
    auto_increment: bool,
    no_seed: bool,
) -> list[int | None]:
    if no_seed:
        return [None] * clip_count

    base = seed if seed is not None else secrets.randbelow(2**31)
    if auto_increment:
        return [base + i for i in range(clip_count)]
    return [base] * clip_count


def concat_clips(clip_paths: list[Path], out_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg required for --concat")

    with _concat_list_file(clip_paths) as list_file:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c", "copy", str(out_path),
            ],
            check=True,
        )


class _concat_list_file:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.list_file: Path | None = None

    def __enter__(self) -> Path:
        import tempfile

        fd, name = tempfile.mkstemp(prefix="h3-concat-", suffix=".txt")
        self.list_file = Path(name)
        lines = []
        for p in self.paths:
            escaped = str(p.resolve()).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        with open(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return self.list_file

    def __exit__(self, *_) -> None:
        if self.list_file is not None:
            self.list_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--clips-dir", type=Path, metavar="DIR",
                        help="Directory with clip .txt files and optional manifest.md")
    parser.add_argument("-p", "--prompt-files", metavar="FILES",
                        help="Comma-separated clip prompt paths (max 8)")
    parser.add_argument("--url", default="http://127.0.0.1:11361", help="mlx-serve base URL")
    parser.add_argument("--ref-image", type=Path, action="append", default=[],
                        metavar="IMAGE", help="Ref2VA ref image (repeatable, max 9)")
    parser.add_argument("--ref-audio", type=Path, action="append", default=[],
                        metavar="AUDIO", help="Ref2VA ref audio (repeatable, max 3)")
    parser.add_argument("--ref-video", type=Path, action="append", default=[],
                        metavar="VIDEO", help="Ref2VA ref video (repeatable, max 3)")
    parser.add_argument(
        "--ref-mod",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="RefMod .safetensors for every clip (repeatable). Not with --ref-image.",
    )
    parser.add_argument("--ref-max-edge", type=int, default=1024, metavar="PX",
                        help="Longest edge for ref images (default 1024; 0 = no resize)")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=416)
    duration_group = parser.add_mutually_exclusive_group()
    duration_group.add_argument("--duration", type=float,
                                help=f"Same length for every clip ({MIN_DURATION_S:g}–{MAX_DURATION_S:g}s)")
    duration_group.add_argument("--total-duration", type=float, metavar="SEC",
                                help="Split total seconds equally across clips (4–15s each)")
    duration_group.add_argument("--frames", type=int,
                                help="Same frame count for every clip (snapped to H3 ladder)")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Diffusion steps (default 30; do not use with --turbo)",
    )
    parser.add_argument(
        "--lora",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="LoRA .safetensors path (repeatable, max 8). Sent as absolute path.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        action="append",
        default=[],
        metavar="F",
        help="Scale per --lora (by index). Omit to use file-declared scales.",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Enable turbo distillation (4 steps; do not pass --steps)",
    )
    parser.add_argument("--seed", type=int, help="Base seed (or with --auto-increment)")
    parser.add_argument("--auto-increment", action="store_true",
                        help="Per-clip seed = base + index")
    parser.add_argument("--no-seed", action="store_true",
                        help="Omit seed; server random per clip")
    parser.add_argument("--no-fast", action="store_true", help="Disable fast recipe")
    parser.add_argument("--concat", action="store_true",
                        help="Concat clips into final.mp4 (off by default)")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help="Run output dir (default: ~/Videos/h3_run_TIMESTAMP)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="HTTP timeout per clip (seconds)")
    parser.add_argument(
        "--continuity",
        choices=sorted(VALID_CONTINUITY),
        help="Default join continuity for clip 2+ when manifest omits per-clip value",
    )
    parser.add_argument(
        "--continuation-tail-seconds",
        type=float,
        metavar="SEC",
        help=f"Prior-clip tail length for video_continuation joins (default {DEFAULT_CONTINUATION_TAIL_S:g})",
    )
    args = parser.parse_args()

    if args.no_seed and (args.seed is not None or args.auto_increment):
        parser.error("--no-seed cannot be used with --seed or --auto-increment")
    if args.turbo and args.steps is not None:
        parser.error("do not pass --steps with --turbo (turbo uses 4 steps)")
    if args.ref_mod and args.ref_image:
        parser.error("do not mix --ref-mod with --ref-image")

    lora_paths = validate_lora_paths(args.lora)
    ref_mod_paths = validate_ref_mod_paths(args.ref_mod) if args.ref_mod else []
    lora_scales = resolve_lora_scales(lora_paths, args.lora_scale)
    steps = resolve_steps(turbo=args.turbo, cli_steps=args.steps)

    manifest: Manifest | None = None
    if args.clips_dir is not None:
        manifest = load_manifest(args.clips_dir.expanduser())

    clip_jobs = resolve_clip_jobs(
        clips_dir=args.clips_dir.expanduser() if args.clips_dir else None,
        prompt_list=args.prompt_files,
        manifest=manifest,
        cli_duration=args.duration,
        cli_total_duration=args.total_duration,
        cli_frames=args.frames,
        cli_continuity=args.continuity,
    )

    continuation_tail_s = resolve_continuation_tail_seconds(
        manifest,
        args.continuation_tail_seconds,
    )

    width = args.width
    height = args.height
    ref_max_edge = args.ref_max_edge
    if manifest:
        if manifest.width is not None:
            width = int(manifest.width)
        if manifest.height is not None:
            height = int(manifest.height)
        if manifest.ref_max_edge is not None:
            ref_max_edge = int(manifest.ref_max_edge)

    width = round32(width)
    height = round32(height)
    if width != args.width or height != args.height:
        print(f"Note: canvas {width}x{height}", file=sys.stderr)

    check_server(args.url)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir or (Path.home() / "Videos" / f"h3_run_{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    ref_paths = prepare_ref_images(args.ref_image, run_dir / "refs", ref_max_edge)

    seeds = resolve_seeds(
        clip_count=len(clip_jobs),
        seed=args.seed,
        auto_increment=args.auto_increment,
        no_seed=args.no_seed,
    )
    if not args.no_seed:
        print(f"Seed policy: base={seeds[0]}", file=sys.stderr)
        if args.auto_increment:
            print(f"  auto-increment through clip {len(clip_jobs)}", file=sys.stderr)

    started_at = datetime.now(timezone.utc).isoformat()
    clip_outputs: list[Path] = []
    meta_clips: list[dict] = []

    for idx, job in enumerate(clip_jobs):
        clip_num = idx + 1
        out_mp4 = run_dir / f"{clip_num:02d}.mp4"
        prompt = job.path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise SystemExit(f"empty prompt file: {job.path}")

        prior_output = clip_outputs[-1] if clip_outputs else None
        ref_videos, continuation_ref = build_ref_videos_for_clip(
            clip_index=idx,
            job=job,
            prior_output=prior_output,
            tail_seconds=continuation_tail_s,
            run_dir=run_dir,
            static_ref_videos=args.ref_video,
        )

        seed = seeds[idx]
        label = f"clip {clip_num}/{len(clip_jobs)} ({job.path.name})"
        cont_note = f", continuity={job.continuity}" if job.continuity else ""
        print(
            f"Generating {label}, {job.duration_s:g}s -> {job.frames} frames, seed={seed}{cont_note} ..."
            if seed is not None
            else f"Generating {label}, {job.duration_s:g}s -> {job.frames} frames{cont_note} ...",
            file=sys.stderr,
        )

        try:
            data = request_video(
                base_url=args.url,
                prompt=prompt,
                width=width,
                height=height,
                num_frames=job.frames,
                steps=steps,
                seed=seed,
                fast=not args.no_fast,
                timeout=args.timeout,
                ref_images=ref_paths or None,
                ref_audios=args.ref_audio or None,
                ref_videos=ref_videos or None,
                ref_mod_paths=ref_mod_paths or None,
                lora_paths=lora_paths or None,
                lora_scales=lora_scales,
                turbo=args.turbo,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise SystemExit(
                f"Clip {clip_num} failed (HTTP {e.code}): {body}\n"
                f"Completed clips kept in {run_dir}"
            ) from e
        except urllib.error.URLError as e:
            raise SystemExit(
                f"Clip {clip_num} failed: {e}\nCompleted clips kept in {run_dir}"
            ) from e

        response_to_mp4(data, out_mp4)
        clip_outputs.append(out_mp4)
        meta_clips.append({
            "index": clip_num,
            "prompt_file": str(job.path.resolve()),
            "output": str(out_mp4.resolve()),
            "duration_seconds": job.duration_s,
            "frames": job.frames,
            "seed": seed,
            "continuity": job.continuity,
            "continuation_ref": str(continuation_ref.resolve()) if continuation_ref else None,
            "continuation_tail_seconds": continuation_tail_s if continuation_ref else None,
            "ref_videos": [str(p.resolve()) for p in ref_videos],
        })
        print(f"Wrote {out_mp4}", file=sys.stderr)

    final_path: Path | None = None
    if args.concat:
        final_path = run_dir / "final.mp4"
        print(f"Concatenating {len(clip_outputs)} clips -> {final_path}", file=sys.stderr)
        concat_clips(clip_outputs, final_path)

    run_meta = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "width": width,
        "height": height,
        "manifest": str((args.clips_dir / MANIFEST_NAME).resolve())
        if args.clips_dir and manifest
        else None,
        "steps": steps,
        "turbo": args.turbo,
        "lora_paths": [str(p) for p in lora_paths],
        "lora_scales": lora_scales,
        "fast": not args.no_fast,
        "seed_base": seeds[0] if seeds and seeds[0] is not None else None,
        "auto_increment": args.auto_increment,
        "no_seed": args.no_seed,
        "ref_images_source": [str(p.resolve()) for p in args.ref_image],
        "ref_images_prepared": [str(p.resolve()) for p in ref_paths],
        "ref_max_edge": ref_max_edge,
        "ref_audios": [str(p.resolve()) for p in args.ref_audio],
        "ref_videos_static": [str(p.resolve()) for p in args.ref_video],
        "ref_mod_paths": [str(p) for p in ref_mod_paths],
        "continuation_tail_seconds": continuation_tail_s,
        "continuity_default": args.continuity,
        "concat": args.concat,
        "final": str(final_path.resolve()) if final_path else None,
        "clips": meta_clips,
    }
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    print(run_dir)
    if final_path is not None:
        print(final_path)


if __name__ == "__main__":
    main()
