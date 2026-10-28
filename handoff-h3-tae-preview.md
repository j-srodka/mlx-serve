# Handoff — mlx-serve: test Draw Things' MiniMax H3 tiny decoder for live previews

Created 2026-10-28 from the h3-timeline session (decision D-59 in
`~/Github/h3-timeline/docs/DECISIONS.md`). Repo for this work: `~/Github/mlx-serve`
(branch `local/ornith-h3` @ `305e53d3`; the working tree already has uncommitted
changes in docs/reference.md, skills/mlx-serve/media.md, src/gen.zig, src/minimax_h3.zig,
lib/mlxc-src — not part of this handoff, leave them alone). Check for drift first.

**Status (2026-10-28): Phase 0 + 0b PASSED.** Core ML ANE warm ~23 ms/chunk at 704²
(artifacts: `~/Github/h3-timeline/.pi-work/stream-preview/PHASE0B_REPORT.md`).
Phase 1 scaffold in mlx-serve: `lib/h3_tae/` + env `MINIMAX_H3_TAE_MODEL` (linear arm:
`MINIMAX_H3_TAE=0`). Shipping still needs your own coremltools build from MIT `taeh3` —
never redistribute Draw Things' zip.

**Was: TEST FIRST.** Phase 0 was a throwaway probe with no server code before Phase 1.

## Why

H3 Timeline streams one preview frame per denoise step (`preview_frames: 1`, D-58)
and, from D-59 on, draws it to FILL the viewer. Today that frame comes from
`src/preview.zig` `latentSliceToRgb`: a per-pixel LINEAR 24→3 projection on the latent
grid, so a 704×704 render previews at **44×44** — a soft blur when upscaled. Draw Things
previews the same model at full resolution with a tiny learned decoder, and the user
wants that quality.

## What Draw Things ships (verified locally, read-only, 2026-10-28)

- Asset: `/Applications/Draw Things.app/Contents/Resources/minimax_h3_tae.zip` (18 MB).
  Contents: `minimax_h3_tae/weight.bin` (19.7 MB fp16, shared) + compiled Core ML models
  `768/1024/1280/1536/1792/2048.mlmodelc` (same weights, per output size).
- Model (1280 variant, from `metadata.json` + `model.mil`): class
  `h3_decoder_stateful_nthwc_chunk2_hw80_dtu000`; coremltools 9.0 from torch 2.6.
  - inputs: `latent` fp16 **[1, 2, 80, 80, 24]** (NTHWC: 2 latent frames per call, latent
    grid padded to 80×80 for 1280 px) + carried state `act_0…act_8` (fp16; 3× [1,80,80,256],
    3× [1,160,160,128], 3× [1,320,320,64]).
  - outputs: `image` fp16 **[1, 2·tUpscale?, 1280, 1280, 3]** in 0…1 (the 1280 build's
    output is [1, 2, 1280, 1280, 3]) + `act_0_out…act_8_out` to feed the next call.
  - first op multiplies the latent by ≈ 1/3 (`0x1.554p-2`) — so WHICH latent space it
    expects (DiT-normalised vs denormalised `z·std+mean`) must be tested, see Phase 0.
  - ~200 conv ops, 3 nearest upsamples + 1 pixel_shuffle → 16× spatial. TAEHV-style
    (madebyollin's tiny video autoencoders): causal, streaming via the carried acts.
- How Draw Things drives it — open-source engine, GPL-3.0
  (https://github.com/drawthingsai/draw-things-community; a clone sits at
  `~/Github/h3-timeline/.pi-work/stream-preview/dtc`, commit 764845c):
  - `Libraries/LocalImageGenerator/Sources/ImageConverter.swift` ~979-1150 (`case .minimaxH3`):
    DT's H3 tensor is NHWC [T, H + audioRows, W, 24]; it drops the audio rows
    (`MiniMaxH3AudioHeight`, `Libraries/SwiftDiffusion/Sources/Models/MiniMaxH3.swift:14`),
    picks the variant by `max(H,W)·16` (≤768 → 768.mlmodelc, …), zero-fills act_0…8, then
    loops `stride(from: 0, to: T, by: chunkT=2)`: copy the chunk into a zero-padded
    [1,2,padH,padW,24] array, `prediction`, carry `act_*_out` → `act_*`, crop each output
    frame to H·16×W·16, clamp `255·x`. It trims the first `tUpscale-1` frames (causal head).
  - `ImageConverter+CoreML.swift` ~530 (unzip + hard-link weight.bin into each .mlmodelc)
    and ~1251-1320 (`MLModelConfiguration.computeUnits = .cpuAndNeuralEngine`).
    **ANE, not GPU** — it doesn't compete with the DiT for the GPU.

## mlx-serve side (facts for the probe)

- Preview hook: `src/minimax_h3.zig` `tryEmitH3Preview` (~4391) → `renderH3Preview`
  computes **x0 = video_x − σ·velocity** (~4431-4436) then the linear projection.
- Saved latents to test with (FL2VA `latent_out` sidecars, real H3 output):
  `~/Github/jake-timeline/takes/<id>/latents.safetensors` →
  `video_latent` F32 **[1, 24, T, H, W]** (e.g. 0683A2D2340B: [1,24,22,44,44] at 704×704,
  next to its `clip.mp4` = ground truth). Whether these are DiT-normalised or not: check
  `src/minimax_h3_vae.zig` ~904-915 (the VAE decode does `z·latents_std + latents_mean`
  first, so the sidecar is most likely the NORMALISED latent).

## Phase 0 — probe (no server changes; ~half a day)

Throwaway probe outside the server build (e.g. `/tmp/h3-tae-probe/`, not wired into zig
build). Primary decoder: **taeh3** (MIT, see below) run through taehv's own reference
`taehv.py` (PyTorch, MPS or CPU) — the ground truth for Phase 1. Optional second arm:
Draw Things' installed Core ML asset (copy the zip to a scratch dir, unzip, hard-link
weight.bin into the chosen .mlmodelc/weights/ as DT does; a Swift CLI) to measure the
ANE speed a Core ML route would get. Never commit or redistribute DT's asset.

1. Load 0683A2D2340B's `video_latent`, transpose [1,24,T,H,W] → [T,H,W,24], fp16, pad
   H/W to the variant's grid (44 → the 768 variant; read the pad size from its
   metadata.json), run the chunk loop exactly as DT does.
2. Latent space: TAEHV decodes the DiT-normalised latent (see below), which is what
   the sidecar should hold — confirm by also trying denormalised `z·std+mean` once; the
   wrong one comes out wrong-coloured/garbage.
3. Compare decoded frames with `clip.mp4` frames (ffmpeg extract): PSNR/SSIM on a few
   frames + a side-by-side PNG sheet for the user.
4. Time it on this Mac: per-call ms for the 768 and 1280 variants on
   `.cpuAndNeuralEngine` vs `.all`; model load time (first call).
5. Preview realism: build an x0-like input from a MID-DENOISE state if possible (or just
   decode the clean latent) and decode ONLY the first chunk (2 latent frames, zero state)
   — that's what a one-frame live preview would cost per step.

**Pass** = recognisable full-res frames close to clip.mp4 (expect soft detail; it's a
tiny decoder), and one-chunk decode ≲ 100 ms on ANE at 704². Report numbers + the sheet.

## Phase 1 — implement (only after the user OKs Phase 0)

Two routes, both on the MIT taeh3 weights; pick from the Phase 0 timings:
(a) **MLX port** of the TAEHV decoder (plain safetensors, ~20 conv blocks — mlx-serve
already loads safetensors and has conv kernels); simplest to ship and portable, but
runs on the GPU between DiT steps. (b) **Core ML** — convert taeh3 ourselves with
coremltools (stateful chunk-2 graph like DT's) and call it through a small Swift/ObjC
shim (C ABI) on `.cpuAndNeuralEngine`, keeping the GPU free for the DiT. Either way:
called from `renderH3Preview` in place of `latentSliceToRgb` when the decoder file is
present (download alongside the pack), linear projection as the fallback. Wire-level: keep the SSE `preview` event
shape; honour `preview_max_side` by downscaling. With a real decoder, the multi-frame
preview (decode a few frames and let the app PLAY them — D-59's "only if it plays")
becomes cheap; and the `preview.zig` 4-frame tile-major/row-major scramble (h3-timeline
LATER.md) is moot for this path.

## Weights + licence — RESOLVED (verified 2026-10-28)

- Public source: **`taeh3.safetensors`** in https://github.com/madebyollin/taehv
  (`safetensors/`, 22,709,752 bytes; 128 tensors, fp16, ~11.3M params; HF mirror
  https://huggingface.co/OzzyGT/taeh3 pins commit 62f7591, sha256
  4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13). taehv README:
  "For **MiniMax H3**, load the `taeh3` weights".
- Licence: **MIT** — "MIT License / Copyright (c) 2025 Ollin Boer Bohan" (taehv
  LICENSE). Shipping/converting is fine with the copyright + permission notice kept.
- Draw Things' `minimax_h3_tae.zip` has no published provenance; it matches taeh3 in
  every observable way (24 ch, ×16 spatial, chunked/stateful, same date range) but that
  is UNCONFIRMED. Don't ship DT's asset; use taeh3 directly. Optional Phase 0 extra:
  compare DT's Core ML output with taeh3's on the same latent.
- Contract (taehv README + FastVideo's minimax_h3_taeh3 docs): TAEHV "does not use any
  latent scales / shifts" — it decodes the **DiT-normalised** latent (NOT `z·std+mean`);
  H3 chunking `5n+2` latent frames → `17n+5` video frames; decode SEQUENTIALLY
  (`parallel=False`) — parallel holds every frame's activations (~25 GB for 209 frames at
  768×1344). Quality: "slightly lower" than the full VAE; much cheaper.
- Lower-quality alternative, not recommended: Kijai/MiniMax-H3-TAE (Apache-2.0).
- Full brief: `~/Github/h3-timeline/.pi-work/stream-preview/RESEARCH_TAE.md`.

