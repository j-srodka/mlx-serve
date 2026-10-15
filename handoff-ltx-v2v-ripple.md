# Handoff — LTX 2.5 V2V / Ripple FFAF feature

Repo: `/Users/jsrodka/Github/mlx-serve` · branch `local/ornith-h3` @ `02bee55` (2026-10-02)

## Summary

Session downloaded assets and scoped a new feature. Nothing in the repo was modified.

**Done this session**
- Downloaded `ddalcu/LTX-2.5-MLX-Serve-8bit` (~60 GiB, the ~69 GB version; 4bit was 38.8 GB) into `~/Models/omlx/ddalcu/LTX-2.5-MLX-Serve-8bit`. mlx-serve discovers it as `ddalcu/LTX-2.5-MLX-Serve-8bit` (video) — needs `--model-dir ~/Models/omlx`. Pack carries both DiTs, gemma4-12b TE, standard VAE decoder AND `vae_diffusion_decoder.safetensors` (the DiffVAE `"decoder":"diffusion"` path).
- Downloaded the LTX-Ripple LoRA (`WepeNerd/LTX-Ripple`, v1.1 weights + config, ComfyUI examples skipped) into `~/Models/LTX-Ripple/LTX25_Ripple_v11.safetensors` (624 MB).
- Extracted Ripple's recommended settings from the model card: LoRA strength **1.35**; default prompt "Use the reference video for motion, timing, camera movement, composition, and unchanged scene content, while consistently propagating the visual edit established in the first frame throughout the video."
- Fetched and traced `Workflow_example/LTX_Ripple_FFAF_Edit_v11.json` from the HF repo — the authoritative conditioning contract (see below, also encoded in the plan).
- Researched the codebase and wrote the implementation plan: **`/Users/jsrodka/.cursor/plans/ltx_v2v_ripple_ffaf_ca68b9b7.plan.md`** (source of truth for the feature — read it first; it is NOT duplicated here).

**Pending**
1. Multi-model adversarial review of the plan (see action item 1).
2. User confirms plan → implement (TDD mandatory per AGENTS.md).

## Priority action items

1. **Run the interrupted `/multi-model-review` of the plan.** The user requested reviewers on model slugs `composer-2.5` and `grok-4.7-high` (their exact words: "Composer 2.5 non-fast and Grok 4.7 High non-fast"; slug disambiguation: composer-2.5 ≠ composer-2.5-fast). Launch both `generalPurpose` Task subagents IN PARALLEL (one message, two Task calls) with identical reviewer prompts: intent paragraph + full plan text + key code excerpts; findings with severity (critical|warning|nit), say "no findings" allowed. Then synthesize: consensus vs lone-model findings, categorize act on / consider / noted / dismissed. NOTE: earlier attempts to spawn Task subagents in that conversation silently failed (no agents appeared); if that recurs, fall back to doing the review directly.
2. **After review + user confirmation, implement the plan** in its todo order. Key contract facts (from the workflow trace): edited first frame + source video frames CONCAT into ONE image sequence → VAE-encoded as one clip → APPENDED as clean in-context tokens (mask 0, 24 fps timeline positions, frame_idx 0, strength 1) beside the generation canvas; audio latent empty (no audio input); CFG off; LoRA 1.35 via existing `lora_paths`.
3. **Watch the VAE-encode transient**: K-frame guide clip encode is a conv3d transient invisible to residency bills (gotcha #321) — chunk if heavy, like `conv3dDepthChunked` does for the decoder.

## Key file paths

- Plan: `/Users/jsrodka/.cursor/plans/ltx_v2v_ripple_ffaf_ca68b9b7.plan.md`
- Engine: `src/ltx_video.zig` — `generateVideoFrames` (~4238), `generateVideoFramesTwoStage` (~4562), `buildKeyframeCond` (~4430), `keyframeMask` (~4397), `keyframePositions` (~4405), `vaeEncode` (~843), `keyframes_abs_pos_embedding` comment (~2728)
- Server: `src/gen.zig` — LTX arm of `handleVideo` (~4080), H3 `ref_videos` parse precedent (~3526), `decodeImageToBCFHW`
- App: `app/Sources/MLXServe/Services/MediaGen.swift` (capabilities), `Views/VideoGenView.swift` (first/last frame sections ~940), `Services/VideoGenService.swift` (`requestBody` ~445-475, `ref_videos` build ~483, AVAssetImageGenerator ~728), `Services/VideoPreprocessor.swift` (`extractFrames` ~31)
- Tests: `tests/test_video_gen.sh`; app `MediaGenServiceTests.swift` / `VideoGenUnreachableSettingsTests.swift`
- Upstream contract artifacts: HF repo `WepeNerd/LTX-Ripple` (`README.md`, `Workflow_example/LTX_Ripple_FFAF_Edit_v11.json` — already fetched once to `/tmp/ripple_wf.json`)

## Architecture context (one-liners)

- LTX conditioning today = first-frame pin (latent replace at frame 0) + last-frame anchor (appended clean token, end-anchored) + a2vid audio (two-stage only). V2V plan generalizes the appended-anchor arm to K frames; `last_frame_image` keeps its own end-anchored semantics, NOT folded into the guide arm.
- Out of scope by design: SDEdit-style denoise-strength V2V (different mechanism, not what Ripple uses) and source-audio conditioning (upstream uses empty audio latent).
- App already has all plumbing: video→frames extraction (client's job per `ChatModels.swift` comment), `ref_videos` body building for H3 ref2va, capability-gated requestBody fields.
- Local-patches rule (`.cursor/rules/mlx-serve-local.mdc`): work on branch `local/ornith-h3`; export ornith/H3-refmod patches before any reset/clean; never delete untracked `generate_h3.py`, `start-minimax-h3*.sh`, `local/patches/`.
- One feature one PR: Zig + app changes land together; commit messages carry no Co-Authored-By.

## What to do next

Run the two parallel reviewer subagents on the plan, synthesize, present the verdict to the user, then (on approval) start the plan's first todo (engine conditioning machinery, red tests first).

## Suggested skills

- `/multi-model-review` — complete the interrupted review (command workflow described in action item 1).
- No other skills needed; follow AGENTS.md TDD rules during implementation.