#!/usr/bin/env python3
"""Quick checks for LoRA/Turbo CLI wiring (no mlx-serve)."""
from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SCRIPTS = [
    ROOT / "generate_h3.py",
    ROOT / "generate_h3_multi.py",
]


def write_png(path: Path, w: int, h: int) -> None:
    raw = b"".join(b"\x00" + (b"\xff\x00\x00" * w) for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    )


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *argv],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def assert_in(needle: str, haystack: str, label: str) -> None:
    if needle not in haystack:
        raise AssertionError(f"{label}: expected {needle!r} in output:\n{haystack}")


def main() -> None:
    for script in SCRIPTS:
        help_out = run([str(script), "--help"])
        if help_out.returncode != 0:
            raise AssertionError(f"{script.name} --help failed:\n{help_out.stderr}")
        for flag in ("--lora", "--lora-scale", "--turbo"):
            assert_in(flag, help_out.stdout, f"{script.name} --help")
        if script.name == "generate_h3.py":
            for flag in ("--ref-mod", "--extract-refmod", "--ref-dir", "--refmod-max-tokens"):
                assert_in(flag, help_out.stdout, f"{script.name} --help")
            assert_in("do NOT use <Picture", help_out.stdout, f"{script.name} --help refmod")
            assert_in("subject_definitions", help_out.stdout, f"{script.name} --help subjects")
            assert_in("2-3 separate", help_out.stdout, f"{script.name} --help stack")
            assert_in("triangulation", help_out.stdout, f"{script.name} --help triangulation")

    turbo_steps = run([str(ROOT / "generate_h3.py"), "--turbo", "--steps", "3", "-p", "x"])
    assert_in("turbo needs at least 4 steps", turbo_steps.stderr + turbo_steps.stdout, "turbo+steps")

    multi_turbo = run([
        str(ROOT / "generate_h3_multi.py"),
        "--turbo",
        "--steps",
        "30",
        "-p",
        "missing.txt",
    ])
    assert_in("do not pass --steps with --turbo", multi_turbo.stderr + multi_turbo.stdout, "multi turbo+steps")

    scale_mismatch = run([
        str(ROOT / "generate_h3.py"),
        "--lora-scale",
        "1.0",
        "-p",
        "x",
    ])
    assert_in("--lora-scale requires at least one --lora", scale_mismatch.stderr + scale_mismatch.stdout, "scale without lora")

    with tempfile.NamedTemporaryFile(suffix=".safetensors") as one:
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as two:
            mismatch = run([
                str(ROOT / "generate_h3.py"),
                "--lora",
                one.name,
                "--lora",
                two.name,
                "--lora-scale",
                "0.5",
                "-p",
                "x",
            ])
    assert_in("must match --lora count", mismatch.stderr + mismatch.stdout, "scale count mismatch")

    missing_lora = run([
        str(ROOT / "generate_h3.py"),
        "--lora",
        "/no/such/lora.safetensors",
        "-p",
        "x",
    ])
    assert_in("LoRA not found", missing_lora.stderr + missing_lora.stdout, "missing lora")

    mix_ref = run([
        str(ROOT / "generate_h3.py"),
        "-p",
        "x",
        "--ref-image",
        "/tmp/a.png",
        "--ref-mod",
        "/tmp/b.safetensors",
    ])
    assert_in("do not mix --ref-mod with --ref-image", mix_ref.stderr + mix_ref.stdout, "ref mix")

    mix_vid = run([
        str(ROOT / "generate_h3.py"),
        "--prompt",
        "x",
        "--ref-video",
        "/tmp/a.mp4",
        "--ref-mod",
        "/tmp/b.safetensors",
    ])
    mix_vid_out = mix_vid.stderr + mix_vid.stdout
    if "do not mix --ref-mod with --ref-image" in mix_vid_out:
        raise AssertionError(f"video mix should be allowed:\n{mix_vid_out}")
    assert_in("RefMod not found", mix_vid_out, "video mix reaches path check")

    mix_aud = run([
        str(ROOT / "generate_h3.py"),
        "--prompt",
        "x",
        "--ref-audio",
        "/tmp/a.wav",
        "--ref-mod",
        "/tmp/b.safetensors",
    ])
    mix_aud_out = mix_aud.stderr + mix_aud.stdout
    if "do not mix --ref-mod with --ref-image" in mix_aud_out:
        raise AssertionError(f"audio mix should be allowed:\n{mix_aud_out}")
    assert_in("RefMod not found", mix_aud_out, "audio mix reaches path check")

    extract_no_img = run([str(ROOT / "generate_h3.py"), "--extract-refmod", "/tmp/out.safetensors"])
    assert_in("extract needs --ref-image/--ref-dir or --ref-video", extract_no_img.stderr + extract_no_img.stdout, "extract needs images")

    extract_visual_audio = run([
        str(ROOT / "generate_h3.py"),
        "--extract-refmod",
        "/tmp/out.safetensors",
        "--ref-audio",
        "/tmp/a.wav",
    ])
    assert_in(
        "--extract-voice-refmod",
        extract_visual_audio.stderr + extract_visual_audio.stdout,
        "visual extract rejects ref-audio",
    )

    voice_no_src = run([str(ROOT / "generate_h3.py"), "--extract-voice-refmod", "/tmp/voice.safetensors"])
    assert_in("exactly one --ref-audio", voice_no_src.stderr + voice_no_src.stdout, "voice needs source")

    voice_mix_visual = run([
        str(ROOT / "generate_h3.py"),
        "--extract-voice-refmod",
        "--ref-audio",
        "/tmp/a.wav",
        "--ref-dir",
        "/tmp",
    ])
    assert_in("do not mix --extract-voice-refmod", voice_mix_visual.stderr + voice_mix_visual.stdout, "voice xor visual")

    extract_xor = run([
        str(ROOT / "generate_h3.py"),
        "--extract-refmod",
        "/tmp/out.safetensors",
        "--ref-dir",
        "/tmp",
        "--ref-video",
        "/tmp/clip.mp4",
    ])
    assert_in("extract stills XOR one --ref-video", extract_xor.stderr + extract_xor.stdout, "extract xor")

    extract_two_vid = run([
        str(ROOT / "generate_h3.py"),
        "--extract-refmod",
        "/tmp/out.safetensors",
        "--ref-video",
        "/tmp/a.mp4",
        "--ref-video",
        "/tmp/b.mp4",
    ])
    assert_in("exactly one --ref-video", extract_two_vid.stderr + extract_two_vid.stdout, "extract one clip")

    extract_prompt = run([
        str(ROOT / "generate_h3.py"),
        "--ref-dir",
        "/tmp",
        "-p",
        "x",
    ])
    assert_in("do not pass --prompt or --prompt-file with RefMod extract", extract_prompt.stderr + extract_prompt.stdout, "extract+prompt")

    from generate_h3 import (
        apply_refmod_token_cap,
        aspect_mismatch,
        default_refmod_out,
        default_refmod_out_video,
        default_refmod_out_voice,
        ensure_safetensors_out,
        linspace_indices,
        merge_extract_images,
        parse_image_size,
        ref_image_canvas,
        resolve_lora_scales,
        resolve_steps,
        tokens_per_latent_frame,
        validate_lora_paths,
        validate_ref_mod_paths,
    )

    assert resolve_steps(turbo=True, cli_steps=None) == 4
    assert resolve_steps(turbo=True, cli_steps=6) == 6
    assert resolve_steps(turbo=False, cli_steps=None) == 30
    assert resolve_steps(turbo=False, cli_steps=20) == 20
    try:
        resolve_steps(turbo=True, cli_steps=3)
        raise AssertionError("expected turbo floor error")
    except SystemExit as e:
        assert "at least 4" in str(e)

    try:
        validate_ref_mod_paths([])
    except SystemExit:
        pass
    try:
        validate_ref_mod_paths([Path("/no/such/ref.safetensors")])
        raise AssertionError("expected missing refmod")
    except SystemExit as e:
        assert "RefMod not found" in str(e)

    assert not aspect_mismatch(100, 100, 100, 100)
    assert not aspect_mismatch(200, 100, 400, 200)
    assert aspect_mismatch(16, 9, 9, 16)

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "crops"
        folder.mkdir()
        cu = folder / "cu.png"
        front = folder / "front.png"
        side = folder / "side.png"
        write_png(cu, 64, 64)
        write_png(front, 128, 72)
        write_png(side, 32, 48)
        hero = folder / "hero.png"
        write_png(hero, 768, 1024)
        (folder / "notes.txt").write_text("skip")
        (folder / "nested").mkdir()
        write_png(folder / "nested" / "hidden.png", 8, 8)

        merged = merge_extract_images([front], folder)
        assert merged[0] == front.resolve()
        assert [p.name for p in merged] == ["front.png", "cu.png", "hero.png", "side.png"]

        assert default_refmod_out(folder) == folder.resolve() / "crops-pack.safetensors"
        clip = Path("/tmp/at-bat.mp4")
        assert default_refmod_out_video(clip) == clip.expanduser().resolve().with_name("at-bat-pack.safetensors")
        wav = Path(tmp) / "line.wav"
        wav.write_bytes(b"x")
        assert default_refmod_out_voice(wav) == wav.resolve().with_name("line-voice.safetensors")

        out = ensure_safetensors_out(Path(tmp) / "new.safetensors")
        assert out.is_file()
        assert parse_image_size(front.read_bytes()) == (128, 72)

        assert linspace_indices(27, 8) == [0, 4, 7, 11, 15, 19, 22, 26]
        assert linspace_indices(4, 4) == [0, 1, 2, 3]
        assert linspace_indices(10, 1) == [0]
        cw, ch = ref_image_canvas(768, 1024, 768, 1024, "match")
        assert (cw, ch) == (768, 1024)
        assert tokens_per_latent_frame(ch // 16, cw // 16) == 768
        copies = [hero] * 27
        kept, dropped, tpf, max_t = apply_refmod_token_cap(
            copies,
            gen_w=768,
            gen_h=1024,
            ref_image_size="match",
            max_tokens=8192,
        )
        assert tpf == 768
        assert max_t == 10
        assert len(kept) == 10
        assert len(dropped) == 17
        assert 0 not in dropped
        cap_vid = run([
            str(ROOT / "generate_h3.py"),
            "--extract-refmod",
            "/tmp/out.safetensors",
            "--ref-video",
            "/tmp/a.mp4",
            "--refmod-max-tokens",
            "8192",
        ])
        assert_in(
            "--refmod-max-tokens applies to stills extract only",
            cap_vid.stderr + cap_vid.stdout,
            "cap xor video",
        )

    check_live_ref_video_clock()
    print("all checks passed")


def check_live_ref_video_clock() -> None:
    """Live ref extract is 24 fps and the wav is cut to the same span."""
    import base64
    import io
    import shutil
    import wave

    if shutil.which("ffmpeg") is None:
        raise AssertionError("ffmpeg required to check ref-video extract")
    from generate_h3 import encode_video_ref

    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.mp4"
        made = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=640x360:rate=30:duration=6",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=16000:duration=6",
                "-shortest",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(clip),
            ],
            capture_output=True,
        )
        if made.returncode != 0:
            raise AssertionError(f"ffmpeg fixture failed:\n{made.stderr.decode(errors='replace')}")
        payload = encode_video_ref(clip, 24, fps=24)
    if len(payload["frames"]) != 24:
        raise AssertionError(f"expected 24 frames, got {len(payload['frames'])}")
    if "audio" not in payload:
        raise AssertionError("live ref dropped the soundtrack")
    with wave.open(io.BytesIO(base64.b64decode(payload["audio"]))) as wav:
        seconds = wav.getnframes() / wav.getframerate()
    if not (0.9 <= seconds <= 1.15):
        raise AssertionError(f"audio should be ~1.0s for 24 frames at 24 fps, got {seconds:.3f}s")


if __name__ == "__main__":
    main()
