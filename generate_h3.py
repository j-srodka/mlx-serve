#!/usr/bin/env python3
"""Call mlx-serve MiniMax-H3 /v1/video/generations and save an mp4.

Pass image/audio/video as local file paths — this script base64-encodes them.
Do not paste base64 yourself.

FL2VA (first/last frame):
  generate_h3.py -p "..." --first-frame shot.png

Ref2VA (multi-reference — needs REF2VA pack + mlx-serve ≥ v26.8.3):
  generate_h3.py -p "<Picture 1> the person walks. overall_soundscape: city." \\
    --ref-image face.png --ref-image outfit.png
  generate_h3.py --prompt-file /abs/prompt.txt --ref-image face.png
  # Server: ./start-minimax-h3-ref2va.sh  (ddalcu MiniMax-H3-REF2VA-MLX-Serve-8bit)

RefMod (extract stills → one .safetensors, then generate with no PNG attach).
Extract (folder glob is non-recursive png/jpg/webp; ASCII-sort leftover files;
explicit --ref-image stay first and set the canvas):
  generate_h3.py --ref-dir ./crops
  generate_h3.py --extract-refmod ./elle-pack.safetensors \\
    --ref-image ./crops/front.png --ref-dir ./crops

  generate_h3.py --extract-refmod --ref-video ./at-bat.mp4
  generate_h3.py --extract-refmod ./at-bat-pack.safetensors --ref-video ./at-bat.mp4

Generate: RefMod skips Qwen — do NOT use <Picture N> on packs. Name the person
in English. Live --ref-video/--ref-audio (mix OK) still use <Video 1>/<Audio 1>.
Do not mix --ref-mod with --ref-image.

Identity stack: same person → 2-3 separate --ref-mod files, not one fat T-stack
(--ref-dir glob is still one video-kind block; no token cap unless --refmod-max-tokens).
Content split, not more images:
  2 packs: close-ups / portraits (face lock)
  1 pack: full body + some faces (wardrobe / stance)
Each file = one DiT cond block. T>1 inside a file is still one fake video.
Folder → one pack stays the extract default; triangulation is N extracts, then
  --ref-mod a --ref-mod b --ref-mod c
Prompt: subject_definitions + name / left-right; no <Picture N> on RefMod.
Strip leftover Picture tags with sed, not grep -v (grep deleted whole lines):
  sed -E 's/<Picture [0-9]+>//g'
  generate_h3.py --prompt "Elle, red hair, close-up portrait. overall_soundscape: quiet room." \
    --ref-mod ./face.safetensors --ref-mod ./body.safetensors --frames 1 -o /tmp/elle.png
  generate_h3.py --prompt "Elle at the plate. <Video 1> is the swing. overall_soundscape: park." \
    --ref-mod ./face.safetensors --ref-mod ./body.safetensors --ref-video ./park.mp4 \
    --seed 847291563 --frames 96 -o /tmp/elle.mp4 --timeout 7200
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
MAX_LORA = 8
ASPECT_WARN_FRAC = 0.01
VAE_SPATIAL = 16
PATCH_H = 2
PATCH_W = 2
REF_IMAGE_SHORT_EDGE = 2048
CANVAS_MULTIPLE = 32


def parse_image_size(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    jpeg = _jpeg_size(data)
    if jpeg:
        return jpeg
    return _webp_size(data)


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(data)
    while i + 8 < n:
        if data[i] != 0xFF:
            return None
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            return None
        marker = data[i]
        i += 1
        if marker in (0xD8, 0xD9, 0x01) or marker == 0x00:
            continue
        if 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > n:
            return None
        seglen = struct.unpack(">H", data[i : i + 2])[0]
        if seglen < 2:
            return None
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 6 > n:
                return None
            h, w = struct.unpack(">HH", data[i + 3 : i + 7])
            return int(w), int(h)
        i += seglen
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    tag = data[12:16]
    if tag == b"VP8X":
        w = 1 + int.from_bytes(data[24:27], "little")
        h = 1 + int.from_bytes(data[27:30], "little")
        return w, h
    if tag == b"VP8 ":
        w = struct.unpack_from("<H", data, 26)[0] & 0x3FFF
        h = struct.unpack_from("<H", data, 28)[0] & 0x3FFF
        return int(w), int(h)
    if tag == b"VP8L" and len(data) >= 25:
        bits = struct.unpack_from("<I", data, 21)[0]
        w = (bits & 0x3FFF) + 1
        h = ((bits >> 14) & 0x3FFF) + 1
        return w, h
    return None


def image_native_size(path: Path) -> tuple[int, int] | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return parse_image_size(data)


def default_refmod_out(ref_dir: Path) -> Path:
    d = ref_dir.expanduser().resolve()
    return d / f"{d.name}-pack.safetensors"


def default_refmod_out_video(clip: Path) -> Path:
    p = clip.expanduser().resolve()
    return p.with_name(f"{p.stem}-pack.safetensors")


def default_refmod_out_voice(media: Path) -> Path:
    p = media.expanduser().resolve()
    return p.with_name(f"{p.stem}-voice.safetensors")


def encode_clip_soundtrack_wav_b64(path: Path) -> str:
    """Extract the soundtrack from a video file as 16 kHz stereo WAV (ref_audios)."""
    if not path.is_file():
        raise SystemExit(f"Video not found: {path}")
    if path.suffix.lower() not in VIDEO_EXTS:
        raise SystemExit(f"Unsupported video type {path.suffix!r}: {path}")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg required for clip soundtrack extract")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "2",
                "-ar",
                "16000",
                str(out),
            ],
            check=True,
        )
        if not out.is_file() or out.stat().st_size < 64:
            raise SystemExit(f"No audio track in {path}")
        return encode_file_b64(out)
    finally:
        out.unlink(missing_ok=True)


def list_ref_dir_images(ref_dir: Path) -> list[Path]:
    d = ref_dir.expanduser().resolve()
    if not d.is_dir():
        raise SystemExit(f"--ref-dir is not a directory: {d}")
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: p.name)
    return files


def merge_extract_images(explicit: list[Path], ref_dir: Path | None) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in explicit:
        path = raw.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Image not found: {path}")
        if path.suffix.lower() not in IMAGE_EXTS:
            raise SystemExit(f"Unsupported image type {path.suffix!r} (use PNG/JPEG/WebP): {path}")
        out.append(path)
        seen.add(path)
    if ref_dir is not None:
        for path in list_ref_dir_images(ref_dir):
            if path in seen:
                continue
            out.append(path)
            seen.add(path)
    if not out:
        raise SystemExit("extract needs at least one image (--ref-image or --ref-dir with png/jpg/webp)")
    return out


def aspect_mismatch(w0: int, h0: int, w: int, h: int, frac: float = ASPECT_WARN_FRAC) -> bool:
    if w0 <= 0 or h0 <= 0 or w <= 0 or h <= 0:
        return False
    a0 = w0 / h0
    return abs((w / h) - a0) / a0 > frac


def ensure_safetensors_out(path: Path) -> Path:
    out = path.expanduser().resolve()
    if out.suffix.lower() != ".safetensors":
        raise SystemExit(f"RefMod out must be .safetensors: {out}")
    parent = out.parent
    if not parent.is_dir():
        raise SystemExit(f"parent directory does not exist: {parent}")
    if out.exists() and out.is_dir():
        raise SystemExit(f"RefMod out is a directory: {out}")
    if not out.exists():
        out.touch()
    return out


def snap32_even(v: float) -> int:
    """Match minimax_h3.snap32 (round-half-even onto a 32 grid)."""
    m = float(CANVAS_MULTIPLE)
    r = round(v / m) * m
    return max(CANVAS_MULTIPLE, int(r))


def ref_image_canvas(
    src_w: int, src_h: int, gen_w: int, gen_h: int, mode: str | None
) -> tuple[int, int]:
    """Client estimate of server refImageCanvas. mode None → match."""
    sizing = "match" if mode is None else mode
    fw, fh = float(src_w), float(src_h)
    if sizing == "max":
        scale = min(1.0, float(REF_IMAGE_SHORT_EDGE) / float(min(src_w, src_h)))
    else:
        scale = min(1.0, (float(gen_w) * float(gen_h) / (fw * fh)) ** 0.5)
    return snap32_even(fw * scale), snap32_even(fh * scale)


def tokens_per_latent_frame(latent_h: int, latent_w: int) -> int:
    return (latent_h // PATCH_H) * (latent_w // PATCH_W)


def linspace_indices(n: int, k: int) -> list[int]:
    """Deterministic subsample along stack order. k >= n → all indices."""
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k <= 0:
        return []
    if k == 1:
        return [0]
    out: list[int] = []
    seen: set[int] = set()
    for i in range(k):
        idx = int(round(i * (n - 1) / (k - 1)))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    if len(out) < k:
        for idx in range(n):
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
                if len(out) == k:
                    break
        out.sort()
    return out


def apply_refmod_token_cap(
    images: list[Path],
    *,
    gen_w: int,
    gen_h: int,
    ref_image_size: str | None,
    max_tokens: int,
) -> tuple[list[Path], list[int], int, int]:
    """Linspace-subsample stills so T * tokens/frame <= max_tokens.

    Returns kept paths, dropped original indices, tokens/frame, max T.
    """
    if max_tokens <= 0:
        raise SystemExit(f"--refmod-max-tokens must be > 0, got {max_tokens}")
    size0 = image_native_size(images[0])
    if size0 is None:
        raise SystemExit(f"cannot estimate token cap: unreadable image [0] {images[0]}")
    cw, ch = ref_image_canvas(size0[0], size0[1], gen_w, gen_h, ref_image_size)
    lh, lw = ch // VAE_SPATIAL, cw // VAE_SPATIAL
    tpf = tokens_per_latent_frame(lh, lw)
    if tpf <= 0:
        raise SystemExit(f"token cap estimate failed (canvas {cw}x{ch})")
    max_t = max(1, max_tokens // tpf)
    kept_idx = linspace_indices(len(images), max_t)
    kept_set = set(kept_idx)
    dropped = [i for i in range(len(images)) if i not in kept_set]
    kept = [images[i] for i in kept_idx]
    return kept, dropped, tpf, max_t


def print_extract_plan(images: list[Path], out_path: Path) -> None:
    print(f"writing {out_path} ({len(images)} images)", file=sys.stderr)
    size0: tuple[int, int] | None = None
    for i, path in enumerate(images):
        size = image_native_size(path)
        if size is None:
            print(f"  [{i}] {path}  (size unknown)", file=sys.stderr)
            continue
        w, h = size
        print(f"  [{i}] {path}  {w}x{h}", file=sys.stderr)
        if i == 0:
            size0 = size
            print(
                f"  canvas from image [0] native {w}x{h} (server snaps; stretch others to that canvas)",
                file=sys.stderr,
            )
            continue
        if size0 is not None and aspect_mismatch(size0[0], size0[1], w, h):
            print(
                f"warning: {path.name} aspect {w}x{h} != image [0] {size0[0]}x{size0[1]}; will stretch",
                file=sys.stderr,
            )


def round32(n: int) -> int:
    return max(32, (n + 31) // 32 * 32)


def snap_frames(n: int) -> int:
    """MiniMax H3 uses a 17k+5 frame ladder; snap up like the server.

    1 is a still (one latent frame, group decode) and must not snap to 5.
    """
    if n == 1:
        return 1
    if n <= 5:
        return 5
    k = (n - 5 + 16) // 17
    return 17 * k + 5


def encode_file_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def encode_image_b64(path: Path) -> str:
    if path.suffix.lower() not in IMAGE_EXTS:
        raise SystemExit(f"Unsupported image type {path.suffix!r} (use PNG/JPEG/WebP): {path}")
    if not path.is_file():
        raise SystemExit(f"Image not found: {path}")
    return encode_file_b64(path)


def encode_audio_wav_b64(path: Path) -> str:
    """Encode audio as WAV bytes (server expects PCM WAV for ref_audios)."""
    if not path.is_file():
        raise SystemExit(f"Audio not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return encode_file_b64(path)
    if suffix not in AUDIO_EXTS:
        raise SystemExit(f"Unsupported audio type {suffix!r}: {path}")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg required to convert non-WAV audio references")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-ac",
                "2",
                "-ar",
                "16000",
                str(out),
            ],
            check=True,
        )
        return encode_file_b64(out)
    finally:
        out.unlink(missing_ok=True)


def encode_video_ref(
    path: Path,
    max_frames: int = 48,
    *,
    include_audio: bool = True,
    fps: int = 8,
) -> dict:
    """Extract frames (+ optional audio) from a video file for ref_videos.

    Live generate passes fps=24 and max_frames=the snapped generation length.
    Audio is cut to that same span. The 8 fps / 48-frame default is RefMod
    extract only. Width stays <=512; the server does not upscale a reference
    video, so a wider extract becomes a bigger VAE encode.
    """
    if fps < 1:
        raise SystemExit(f"ref video fps must be >= 1, got {fps}")
    if max_frames < 5:
        raise SystemExit(f"ref video needs at least 5 frames, got {max_frames}")
    if not path.is_file():
        raise SystemExit(f"Video not found: {path}")
    if path.suffix.lower() not in VIDEO_EXTS:
        raise SystemExit(f"Unsupported video type {path.suffix!r}: {path}")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg required for --ref-video")

    duration = max_frames / fps
    with tempfile.TemporaryDirectory(prefix="h3-refvid-") as tmp:
        tmpdir = Path(tmp)
        pattern = tmpdir / "frame_%04d.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vf",
                f"fps={fps},scale='min(512,iw)':-2",
                "-frames:v",
                str(max_frames),
                str(pattern),
            ],
            check=True,
        )
        frames = sorted(tmpdir.glob("frame_*.png"))
        if len(frames) < 5:
            raise SystemExit(
                f"Video ref needs at least 5 frames after extract; got {len(frames)} from {path}"
            )
        payload: dict = {"frames": [encode_file_b64(f) for f in frames]}
        print(
            f"ref video {path.name}: {len(frames)} frames @ {fps} fps"
            + (f", audio {duration:.3f}s" if include_audio else ", no audio"),
            file=sys.stderr,
        )
        if not include_audio:
            return payload

        wav = tmpdir / "audio.wav"
        aud = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "2",
                "-ar",
                "16000",
                "-t",
                f"{duration:.6f}",
                str(wav),
            ],
            capture_output=True,
        )
        if aud.returncode == 0 and wav.is_file() and wav.stat().st_size > 64:
            payload["audio"] = encode_file_b64(wav)
        return payload


def validate_ref_mod_paths(paths: list[Path]) -> list[Path]:
    if len(paths) > 12:
        raise SystemExit("at most 12 --ref-mod files")
    resolved: list[Path] = []
    for path in paths:
        abs_path = path.expanduser().resolve()
        if not abs_path.is_file():
            raise SystemExit(f"RefMod not found: {abs_path}")
        if abs_path.suffix.lower() != ".safetensors":
            raise SystemExit(f"RefMod must be .safetensors: {abs_path}")
        resolved.append(abs_path)
    return resolved


def request_extract_refmod(
    *,
    base_url: str,
    out_path: Path,
    ref_images: list[Path] | None = None,
    ref_videos: list[dict] | None = None,
    width: int,
    height: int,
    ref_image_size: str | None,
    timeout: int,
) -> dict:
    if bool(ref_images) == bool(ref_videos):
        raise SystemExit("extract needs stills XOR one ref_videos payload")
    body: dict = {
        "out_path": str(out_path.expanduser().resolve()),
        "width": width,
        "height": height,
    }
    if ref_images:
        body["ref_images"] = [encode_image_b64(p) for p in ref_images]
        if ref_image_size is not None:
            if ref_image_size not in ("match", "max"):
                raise SystemExit("--ref-image-size must be 'match' or 'max'")
            body["ref_image_size"] = ref_image_size
    else:
        body["ref_videos"] = ref_videos
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/video/refmods",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def request_extract_voice_refmod(
    *,
    base_url: str,
    out_path: Path,
    ref_audio_b64: str,
    voice_max_seconds: float,
    timeout: int,
) -> dict:
    if voice_max_seconds <= 0:
        raise SystemExit("--voice-max-seconds must be positive")
    body: dict = {
        "out_path": str(out_path.expanduser().resolve()),
        "ref_audios": [ref_audio_b64],
        "voice_max_seconds": voice_max_seconds,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/video/refmods",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def validate_lora_paths(paths: list[Path]) -> list[Path]:
    if len(paths) > MAX_LORA:
        raise SystemExit(f"at most {MAX_LORA} --lora files")
    resolved: list[Path] = []
    for path in paths:
        abs_path = path.expanduser().resolve()
        if not abs_path.is_file():
            raise SystemExit(f"LoRA not found: {abs_path}")
        if abs_path.suffix.lower() != ".safetensors":
            raise SystemExit(f"LoRA must be .safetensors: {abs_path}")
        resolved.append(abs_path)
    return resolved


def resolve_lora_scales(lora_paths: list[Path], scales: list[float]) -> list[float] | None:
    if scales and not lora_paths:
        raise SystemExit("--lora-scale requires at least one --lora")
    if not scales:
        return None
    if len(scales) != len(lora_paths):
        raise SystemExit(
            f"--lora-scale count ({len(scales)}) must match --lora count ({len(lora_paths)})"
        )
    return scales


def resolve_prompt(prompt: str | None, prompt_file: Path | None) -> str | None:
    """Return prompt text. --prompt and --prompt-file are mutually exclusive."""
    if prompt_file is None:
        return prompt
    path = prompt_file.expanduser()
    if not path.is_file():
        raise SystemExit(f"prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"prompt file is empty: {path}")
    return text


def resolve_steps(*, turbo: bool, cli_steps: int | None) -> int:
    """Turbo default is 4. The server accepts any turbo step count >= 4."""
    if turbo:
        if cli_steps is None:
            return 4
        if cli_steps < 4:
            raise SystemExit("turbo needs at least 4 steps")
        return cli_steps
    return cli_steps if cli_steps is not None else 30


DEFAULT_AB_LATENT_PREFIX_FRAMES = 22


def request_video(
    *,
    base_url: str,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    seed: int | None,
    fast: bool,
    timeout: int,
    first_frame: Path | None = None,
    last_frame: Path | None = None,
    ref_images: list[Path] | None = None,
    ref_audios: list[Path] | None = None,
    ref_videos: list[Path] | None = None,
    ref_image_size: str | None = None,
    ref_mod_paths: list[Path] | None = None,
    lora_paths: list[Path] | None = None,
    lora_scales: list[float] | None = None,
    turbo: bool = False,
    chain_windows: int = 1,
    latent_prefix_frames: int = 0,
    latent_out: str | None = None,
    latent_prefix_path: str | None = None,
) -> dict:
    body: dict = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "steps": steps,
        "fast": fast,
    }
    if seed is not None:
        body["seed"] = seed
    if first_frame is not None:
        body["first_frame_image"] = encode_image_b64(first_frame)
    if last_frame is not None:
        body["last_frame_image"] = encode_image_b64(last_frame)

    if ref_images:
        if len(ref_images) > 9:
            raise SystemExit("at most 9 --ref-image files")
        body["ref_images"] = [encode_image_b64(p) for p in ref_images]
        print(f"Encoded {len(ref_images)} ref image(s)", file=sys.stderr)
        if ref_image_size is not None:
            if ref_image_size not in ("match", "max"):
                raise SystemExit("--ref-image-size must be 'match' or 'max'")
            body["ref_image_size"] = ref_image_size
    if ref_audios:
        if len(ref_audios) > 3:
            raise SystemExit("at most 3 --ref-audio files")
        body["ref_audios"] = [encode_audio_wav_b64(p) for p in ref_audios]
        print(f"Encoded {len(ref_audios)} ref audio(s)", file=sys.stderr)
    if ref_videos:
        if len(ref_videos) > 3:
            raise SystemExit("at most 3 --ref-video files")
        body["ref_videos"] = [
            encode_video_ref(p, num_frames, fps=24) for p in ref_videos
        ]
        print(f"Encoded {len(ref_videos)} ref video(s)", file=sys.stderr)

    n_files = (
        len(ref_images or [])
        + len(ref_audios or [])
        + len(ref_videos or [])
        + len(ref_mod_paths or [])
    )
    if n_files > 12:
        raise SystemExit("at most 12 reference files total across images/videos/audios/ref-mod")

    if ref_mod_paths:
        body["ref_mod_paths"] = [str(p) for p in ref_mod_paths]
        print(f"Using {len(ref_mod_paths)} ref mod file(s)", file=sys.stderr)
        for i, p in enumerate(ref_mod_paths):
            tag = "identity/first" if i == 0 else f"slot {i}"
            print(f"  ref-mod[{i}] ({tag}): {p}", file=sys.stderr)

    if lora_paths:
        body["lora_paths"] = [str(p) for p in lora_paths]
        if lora_scales:
            body["lora_scales"] = lora_scales
    if turbo:
        body["turbo"] = True
    if chain_windows > 1:
        body["chain_windows"] = chain_windows
    if latent_prefix_frames != 0:
        body["latent_prefix_frames"] = latent_prefix_frames
    if latent_out:
        body["latent_out"] = latent_out
    if latent_prefix_path:
        body["latent_prefix_path"] = latent_prefix_path

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/video/generations",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def load_response(path: str | None) -> dict:
    if path and path != "-":
        return json.loads(Path(path).read_text())
    return json.loads(sys.stdin.read())


def response_to_mp4(data: dict, out_path: Path) -> Path:
    if data.get("format") != "rgb8":
        raise SystemExit(f"Unexpected frame format: {data.get('format')!r}")

    frames = int(data["frames"])
    width = int(data["width"])
    height = int(data["height"])
    fps = int(data.get("fps", 24))
    rgb = base64.b64decode(data["data"])
    expected = frames * width * height * 3
    if len(rgb) != expected:
        raise SystemExit(f"RGB size mismatch: got {len(rgb)}, expected {expected}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="h3-gen-") as tmp:
        tmpdir = Path(tmp)
        for i in range(frames):
            frame_path = tmpdir / f"frame_{i:04d}.ppm"
            with open(frame_path, "wb") as f:
                f.write(f"P6\n{width} {height}\n255\n".encode())
                f.write(rgb[i * width * height * 3 : (i + 1) * width * height * 3])

        video_only = tmpdir / "video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(tmpdir / "frame_%04d.ppm"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video_only),
            ],
            check=True,
        )

        audio_b64 = data.get("audio_data")
        if audio_b64:
            pcm = base64.b64decode(audio_b64)
            wav_path = tmpdir / "audio.wav"
            sr = int(data.get("audio_sample_rate", 32000))
            ch = int(data.get("audio_channels", 2))
            data_size = len(pcm)
            with open(wav_path, "wb") as f:
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + data_size))
                f.write(b"WAVEfmt ")
                f.write(struct.pack("<IHHIIHH", 16, 1, ch, sr, sr * ch * 2, ch * 2, 16))
                f.write(b"data")
                f.write(struct.pack("<I", data_size))
                f.write(pcm)

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video_only),
                    "-i",
                    str(wav_path),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(out_path),
                ],
                check=True,
            )
        else:
            shutil.copy(video_only, out_path)


def response_to_png(data: dict, out_path: Path) -> Path:
    """Write frame 0 of an rgb8 stills response as PNG via ffmpeg."""
    if data.get("format") != "rgb8":
        raise SystemExit(f"Unexpected frame format: {data.get('format')!r}")
    frames = int(data["frames"])
    width = int(data["width"])
    height = int(data["height"])
    rgb = base64.b64decode(data["data"])
    expected = frames * width * height * 3
    if len(rgb) != expected:
        raise SystemExit(f"RGB size mismatch: got {len(rgb)}, expected {expected}")
    if frames < 1:
        raise SystemExit("stills response has no frames")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ppm = (
        f"P6\n{width} {height}\n255\n".encode()
        + rgb[0 : width * height * 3]
    )
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as tmp:
        tmp.write(ppm)
        ppm_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(ppm_path),
                str(out_path),
            ],
            check=True,
        )
    finally:
        ppm_path.unlink(missing_ok=True)
    return out_path


def extract_ab_join_strips(
    arm_a_mp4: Path,
    arm_b_mp4: Path,
    out_dir: Path,
    join_frame_index: int,
) -> None:
    """Pull 16 frames (8 before + 8 after) around the window join on both arms."""
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg required for --ab join frame extract")
    w = join_frame_index
    select = f"select='between(n,{w - 8},{w + 7})'"
    for label, src in (("ab-a", arm_a_mp4), ("ab-b", arm_b_mp4)):
        strip_dir = out_dir / label
        strip_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vf",
                select,
                "-fps_mode",
                "passthrough",
                str(strip_dir / "f%03d.png"),
            ],
            check=True,
        )


def run_latent_prefix_ab(
    *,
    base_url: str,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    seed: int | None,
    timeout: int,
    out_dir: Path,
    chain_windows: int,
    arm_b_latent_prefix_frames: int,
    lora_paths: list[Path] | None = None,
    lora_scales: list[float] | None = None,
) -> None:
    """FL2VA A/B: pixel chain (prefix 0) vs latent pin (prefix N). fast/turbo off."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_a = out_dir / "arm-a.mp4"
    arm_b = out_dir / "arm-b.mp4"
    common = {
        "base_url": base_url,
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "steps": steps,
        "seed": seed,
        "fast": False,
        "timeout": timeout,
        "chain_windows": chain_windows,
        "turbo": False,
        "lora_paths": lora_paths,
        "lora_scales": lora_scales,
    }
    print("A/B arm A: pixel chain (latent_prefix_frames=0)", file=sys.stderr)
    data_a = request_video(**common, latent_prefix_frames=0)
    response_to_mp4(data_a, arm_a)
    print(f"Wrote {arm_a}", file=sys.stderr)

    print(
        f"A/B arm B: latent pin (latent_prefix_frames={arm_b_latent_prefix_frames})",
        file=sys.stderr,
    )
    data_b = request_video(**common, latent_prefix_frames=arm_b_latent_prefix_frames)
    response_to_mp4(data_b, arm_b)
    print(f"Wrote {arm_b}", file=sys.stderr)

    align_msg = f"aligned join frame index W={num_frames} (per-window snapped frame count)"
    print(align_msg, file=sys.stderr)
    print(align_msg)
    extract_ab_join_strips(arm_a, arm_b, out_dir, num_frames)
    print(f"Join strips under {out_dir / 'ab-a'} and {out_dir / 'ab-b'}", file=sys.stderr)
    print(out_dir)


def main() -> None:
    default_out = Path.home() / "Videos"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from-json",
        metavar="FILE",
        help="Convert an existing API JSON response to mp4 (use - for stdin)",
    )
    parser.add_argument("--url", default="http://127.0.0.1:11361", help="mlx-serve base URL")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("-p", "--prompt", help="Generation prompt")
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        metavar="FILE",
        help="Read the generation prompt from a UTF-8 text file (mutually exclusive with --prompt)",
    )
    parser.add_argument(
        "--first-frame",
        type=Path,
        metavar="IMAGE",
        help="FL2VA first-frame image path (PNG/JPEG/WebP)",
    )
    parser.add_argument(
        "--last-frame",
        type=Path,
        metavar="IMAGE",
        help="FL2VA last-frame image path (PNG/JPEG/WebP)",
    )
    parser.add_argument(
        "--ref-image",
        type=Path,
        action="append",
        default=[],
        metavar="IMAGE",
        help="Ref2VA reference image path (repeatable, max 9). Script base64-encodes it.",
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        action="append",
        default=[],
        metavar="AUDIO",
        help="Ref2VA reference audio path (repeatable, max 3). WAV preferred; others via ffmpeg.",
    )
    parser.add_argument(
        "--ref-video",
        type=Path,
        action="append",
        default=[],
        metavar="VIDEO",
        help="Ref2VA reference video path (repeatable, max 3). With --extract-refmod: exactly one clip → one pack (no audio). Without extract: live generate.",
    )
    parser.add_argument(
        "--ref-image-size",
        choices=("match", "max"),
        help="Ref2VA reference token resolution: match generation size or max detail (slower).",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        metavar="DIR",
        help="Folder of stills for RefMod extract (non-recursive png/jpg/webp). Implies extract; default OUT is DIR/<name>-pack.safetensors. Every matching file is included unless --refmod-max-tokens; extras like foo-5mp.png are not skipped; print lists them.",
    )
    parser.add_argument(
        "--extract-refmod",
        nargs="?",
        const="__default__",
        default=None,
        metavar="OUT.safetensors",
        help="Extract RefMod pack (POST /v1/video/refmods). Stills: --ref-image/--ref-dir. Clip: exactly one --ref-video. OUT optional: DIR/<name>-pack.safetensors or clip-pack.safetensors beside the file.",
    )
    parser.add_argument(
        "--extract-voice-refmod",
        nargs="?",
        const="__default__",
        default=None,
        metavar="OUT-voice.safetensors",
        help="Extract a character voice pack (audio VAE latent). Exactly one --ref-audio or one --ref-video soundtrack. Default OUT: <stem>-voice.safetensors beside the source file.",
    )
    parser.add_argument(
        "--voice-max-seconds",
        type=float,
        default=8.0,
        metavar="SEC",
        help="Voice extract only: keep this many seconds from the start (default 8).",
    )
    parser.add_argument(
        "--refmod-max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Stills extract only: linspace-subsample stack order so T * tokens/frame <= N (Malcolm 8192). Logs source_frames vs latent_t and dropped indices. Default: no cap — --ref-dir still encodes every file.",
    )
    parser.add_argument(
        "--ref-mod",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="RefMod .safetensors for generate (repeatable). Mix OK with --ref-video/--ref-audio; not with --ref-image.",
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=416)
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Source frames. 1 is a still (one latent frame, group decode, PNG; does not snap to 5).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Diffusion steps (default 30; with --turbo default 4, minimum 4)",
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
        help="Enable turbo distillation (default 4 steps, minimum 4)",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-fast", action="store_true", help="Disable fast recipe")
    parser.add_argument(
        "--chain-windows",
        type=int,
        default=1,
        metavar="N",
        help="FL2VA chained windows (2–6). Sent when N > 1.",
    )
    parser.add_argument(
        "--latent-prefix-frames",
        type=int,
        default=0,
        metavar="N",
        help="FL2VA latent-tail overlap (e.g. 22). Requires --chain-windows > 1. "
        "With --ab, sets arm B (default 22); arm A stays 0.",
    )
    parser.add_argument(
        "--ab",
        action="store_true",
        help="FL2VA A/B: arm-a.mp4 (pixel chain) then arm-b.mp4 (latent pin) under -o; "
        "forces fast/turbo off; extracts join frame strips at snapped --frames.",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output path (mp4, or png when --frames 1)")
    parser.add_argument("--timeout", type=int, default=3600, help="HTTP timeout seconds")
    args = parser.parse_args()

    if args.from_json:
        data = load_response(args.from_json)
    elif args.extract_voice_refmod is not None:
        if args.prompt or args.prompt_file:
            parser.error("do not pass --prompt or --prompt-file with voice RefMod extract")
        if args.ref_mod:
            parser.error("do not pass --ref-mod with extract")
        if args.extract_refmod is not None or args.ref_dir is not None or args.ref_image:
            parser.error("do not mix --extract-voice-refmod with visual extract (--ref-dir/--ref-image/--extract-refmod)")
        if args.refmod_max_tokens is not None:
            parser.error("--refmod-max-tokens applies to stills visual extract only")
        n_aud = len(args.ref_audio)
        n_vid = len(args.ref_video)
        if n_aud + n_vid != 1:
            parser.error("voice extract needs exactly one --ref-audio or one --ref-video (soundtrack)")
        if n_aud == 1:
            src = args.ref_audio[0]
            audio_b64 = encode_audio_wav_b64(src)
            if args.extract_voice_refmod == "__default__":
                out_path = default_refmod_out_voice(src)
            else:
                out_path = Path(args.extract_voice_refmod)
            print(f"voice source: audio file {src}", file=sys.stderr)
        else:
            src = args.ref_video[0]
            audio_b64 = encode_clip_soundtrack_wav_b64(src)
            if args.extract_voice_refmod == "__default__":
                out_path = default_refmod_out_voice(src)
            else:
                out_path = Path(args.extract_voice_refmod)
            print(f"voice source: clip soundtrack {src}", file=sys.stderr)
        out_path = ensure_safetensors_out(out_path)
        print(
            f"writing {out_path} (first {args.voice_max_seconds}s, kind=audio)",
            file=sys.stderr,
        )
        try:
            data = request_extract_voice_refmod(
                base_url=args.url,
                out_path=out_path,
                ref_audio_b64=audio_b64,
                voice_max_seconds=args.voice_max_seconds,
                timeout=args.timeout,
            )
        except SystemExit as e:
            parser.error(str(e))
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise SystemExit(f"HTTP {e.code}: {body}") from e
        print(json.dumps(data, indent=2))
        return
    elif args.extract_refmod is not None or args.ref_dir is not None:
        if args.prompt or args.prompt_file:
            parser.error("do not pass --prompt or --prompt-file with RefMod extract")
        if args.ref_mod:
            parser.error("do not pass --ref-mod with extract")
        if args.ref_audio:
            parser.error("visual extract does not store audio (use --extract-voice-refmod for voice packs)")
        n_vid = len(args.ref_video)
        has_stills = bool(args.ref_image or args.ref_dir)
        if n_vid and has_stills:
            parser.error("extract stills XOR one --ref-video (do not mix)")
        if n_vid > 1:
            parser.error("extract takes exactly one --ref-video")
        if n_vid == 0 and not has_stills:
            parser.error("extract needs --ref-image/--ref-dir or --ref-video")
        width = round32(args.width)
        height = round32(args.height)
        try:
            if n_vid == 1:
                clip = args.ref_video[0]
                if args.extract_refmod is None or args.extract_refmod == "__default__":
                    out_path = default_refmod_out_video(clip)
                else:
                    out_path = Path(args.extract_refmod)
                out_path = ensure_safetensors_out(out_path)
                if args.refmod_max_tokens is not None:
                    parser.error("--refmod-max-tokens applies to stills extract only (not --ref-video)")
                payload = encode_video_ref(clip, include_audio=False, fps=8)
                n_frames = len(payload["frames"])
                print(
                    f"writing {out_path} ({n_frames} frames from {clip}, no audio)",
                    file=sys.stderr,
                )
                data = request_extract_refmod(
                    base_url=args.url,
                    out_path=out_path,
                    ref_videos=[payload],
                    width=width,
                    height=height,
                    ref_image_size=args.ref_image_size,
                    timeout=args.timeout,
                )
            else:
                images = merge_extract_images(args.ref_image, args.ref_dir)
                if args.extract_refmod is None or args.extract_refmod == "__default__":
                    if args.ref_dir is None:
                        parser.error("stills extract needs --extract-refmod OUT or --ref-dir")
                    out_path = default_refmod_out(args.ref_dir)
                else:
                    out_path = Path(args.extract_refmod)
                out_path = ensure_safetensors_out(out_path)
                source_frames = len(images)
                dropped: list[int] = []
                if args.refmod_max_tokens is not None:
                    images, dropped, tpf, max_t = apply_refmod_token_cap(
                        images,
                        gen_w=width,
                        gen_h=height,
                        ref_image_size=args.ref_image_size,
                        max_tokens=args.refmod_max_tokens,
                    )
                    print(
                        f"token cap {args.refmod_max_tokens}: source_frames={source_frames} "
                        f"kept={len(images)} max_t={max_t} tokens/frame~{tpf} dropped={dropped}",
                        file=sys.stderr,
                    )
                print_extract_plan(images, out_path)
                data = request_extract_refmod(
                    base_url=args.url,
                    out_path=out_path,
                    ref_images=images,
                    width=width,
                    height=height,
                    ref_image_size=args.ref_image_size,
                    timeout=args.timeout,
                )
                latent_t = data.get("latent_t")
                print(
                    f"source_frames={source_frames} latent_t={latent_t} sent={len(images)}",
                    file=sys.stderr,
                )
        except SystemExit as e:
            parser.error(str(e))
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise SystemExit(f"HTTP {e.code}: {body}") from e
        print(json.dumps(data, indent=2))
        return
    else:
        prompt = resolve_prompt(args.prompt, args.prompt_file)
        if not prompt:
            parser.error("provide --prompt, --prompt-file, or --from-json")
        if (args.first_frame or args.last_frame) and (
            args.ref_image or args.ref_audio or args.ref_video or args.ref_mod
        ):
            parser.error("do not mix FL2VA (--first-frame/--last-frame) with Ref2VA (--ref-*)")
        if args.ref_mod and args.ref_image:
            parser.error("do not mix --ref-mod with --ref-image")
        if args.turbo and args.steps is not None and args.steps < 4:
            parser.error("turbo needs at least 4 steps")
        if args.chain_windows < 1 or args.chain_windows > 6:
            parser.error("--chain-windows must be 1–6")
        if args.latent_prefix_frames != 0 and args.chain_windows < 2:
            parser.error("--latent-prefix-frames requires --chain-windows > 1")
        if args.ab:
            if args.from_json:
                parser.error("do not use --from-json with --ab")
            if args.chain_windows < 2:
                parser.error("--ab requires --chain-windows >= 2 (FL2VA latent chain)")
            if not args.output:
                parser.error("--ab requires -o OUTPUT_DIR")
            if (
                args.first_frame
                or args.last_frame
                or args.ref_image
                or args.ref_audio
                or args.ref_video
                or args.ref_mod
            ):
                parser.error(
                    "--ab is FL2VA-only: no --first-frame/--last-frame or --ref-*"
                )

        lora_paths = validate_lora_paths(args.lora)
        ref_mod_paths = validate_ref_mod_paths(args.ref_mod) if args.ref_mod else []
        lora_scales = resolve_lora_scales(lora_paths, args.lora_scale)
        steps = resolve_steps(turbo=args.turbo, cli_steps=args.steps)

        width = round32(args.width)
        height = round32(args.height)
        if width != args.width or height != args.height:
            print(
                f"Note: rounded size {args.width}x{args.height} -> {width}x{height} (must be multiples of 32)",
                file=sys.stderr,
            )
        frames = snap_frames(args.frames)
        if frames != args.frames:
            print(f"Note: frame count may snap to {frames} on the server", file=sys.stderr)
        if args.ab:
            ab_b_prefix = (
                args.latent_prefix_frames
                if args.latent_prefix_frames != 0
                else DEFAULT_AB_LATENT_PREFIX_FRAMES
            )
            ab_steps = args.steps if args.steps is not None else 30
            try:
                run_latent_prefix_ab(
                    base_url=args.url,
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_frames=frames,
                    steps=ab_steps,
                    seed=args.seed,
                    timeout=args.timeout,
                    out_dir=args.output.expanduser().resolve(),
                    chain_windows=args.chain_windows,
                    arm_b_latent_prefix_frames=ab_b_prefix,
                    lora_paths=lora_paths or None,
                    lora_scales=lora_scales,
                )
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                raise SystemExit(f"HTTP {e.code}: {body}") from e
            return

        try:
            data = request_video(
                base_url=args.url,
                prompt=prompt,
                width=width,
                height=height,
                num_frames=frames,
                steps=steps,
                seed=args.seed,
                fast=not args.no_fast,
                timeout=args.timeout,
                first_frame=args.first_frame,
                last_frame=args.last_frame,
                ref_images=args.ref_image or None,
                ref_audios=args.ref_audio or None,
                ref_videos=args.ref_video or None,
                ref_mod_paths=ref_mod_paths or None,
                ref_image_size=args.ref_image_size,
                lora_paths=lora_paths or None,
                lora_scales=lora_scales,
                turbo=args.turbo,
                chain_windows=args.chain_windows,
                latent_prefix_frames=args.latent_prefix_frames,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise SystemExit(f"HTTP {e.code}: {body}") from e

    stills = int(data.get("frames", 0)) == 1 or args.frames == 1
    if args.output:
        out = args.output
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "png" if stills else "mp4"
        out = default_out / f"minimax_h3_{stamp}.{ext}"

    if stills and out.suffix.lower() != ".mp4":
        path = response_to_png(data, out if out.suffix.lower() == ".png" else out.with_suffix(".png"))
    else:
        path = response_to_mp4(data, out)
    print(path)


if __name__ == "__main__":
    main()
