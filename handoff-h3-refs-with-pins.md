# Handoff: mlx-serve, let H3 references ride pinned windows (Phase B)

**Status:** Phase B steps 1–6 landed 2026-11-08 (`local/ornith-h3`). GPU probe
remains h3-timeline; downstream D-80/D-82 still open.
**Predecessor:** `docs/archive/handoff-ref2va-latent-out.md`. Its Phase A
(`latent_out` with refs) shipped; **Phase B (pins with refs) was left open.**
This is Phase B.
**Branch at writing:** `local/ornith-h3` @ `1b5e0647`.

## Why now (evidence from h3-timeline, 2026-11-07/08)
A 60 s tracking shot of a walking man, rendered as a Long take (11 windows of
`chain_windows: 1`, each pinned to the last 22 frames of the previous one):
- **No anchor:** the man's identity drifts. His short, neat hair turns curly,
  his face and beard change, and the picture gets harsher by the end. The
  user saw it clearly.
- **Hidden keystone** (segment 1's frame-0 latent in each prefix): helps, but
  not enough (the user's verdict).
- **Hidden capstone/both:** loops the scene on moving shots. Unusable.
- **Root limit:** the person's references can only ride segment 1, because
  `refCompatError` refuses refs with `latent_prefix_frames` /
  `latent_suffix_frames`. Every later window learns who he is only from 22
  held frames of an already-drifted copy.

## Where the refusal lives
- `src/plugins/minimax_h3/h3.zig` ~4257-4273 `refCompatError` (named 400s for
  chain / prefix / suffix with refs), with the test at ~8050.
- `src/plugins/minimax_h3/plugin.zig:513`, the call site.
- The layout already supports both segment kinds in one sequence: `.ref_img`
  and `.video_pin` segments are both built in the same layout pass (h3.zig
  ~650-780), and both get the conditioning timestep in `buildTimestepPlan`
  (h3.zig ~1891/1919, as fixed in `b9e6e82f`). So this may be mostly lifting
  a policy refusal, but that has to be proven, not assumed.

## Plan (smallest first)
1. **Read before changing (no GPU):** confirm, from the layout code, the
   order and RoPE time placement of `ref_img` / `ref_audio` rows against
   `video_pin` / `audio_pin` rows when both are present. Refs are placed
   outside the video time grid; pins sit on the grid at the window's head and
   tail. Write the answer into this file. **If the two collide** (same
   positions, overlapping rows, or a mask that assumes one or the other),
   stop and report: the fix becomes a layout change, not a refusal lift.
2. **Lift the refusal for prefix/suffix pins only,** on the hybrid FL2VA+REF2VA
   pack (`e.supports_fl2va` and refs supported). Keep `chain_windows > 1` +
   refs refused. Keep a named 400 on packs without FL2VA.
3. **Memory:** refs add rows to every pinned window. Fold `ref_img_rows` /
   `ref_audio_rows` into the existing per-request memory estimate (the
   "needs ~X GB" refusal), so an over-budget window refuses by name, never by
   OOM.
4. **Advertise it:** `GET /health` gains `"ref_pins": true` (beside
   `latent_suffix` and `ref_latent_out`), so the app gates without trial and
   error.
5. **Tests (hermetic):** update the `refCompatError` matrix; add a layout test
   with refs + prefix pin + suffix pin, asserting segment kinds, row counts,
   no overlap, and the conditioning timesteps on both ref and pin rows.
6. **Docs:** `docs/reference.md` request surface, `skills/mlx-serve/media.md`,
   per this repo's AGENTS.md (small diffs, sidecar export before/after a
   parent commit).

## Step 1 verdict (2026-11-08, read `PackedLayout.initFull`)

**No layout collision — policy lift is safe for video refs + video/audio pins.**

Packed order is always `text → fl2va cond → ref2va blocks (`.ref_img` /
`.ref_audio`) → target audio (`.audio_pin` | `.audio` | `.audio_pin`) → target
video (`.video_pin` | `.video` | `.video_pin`)`. References advance a
`cursor` on their own timeline *before* target streams; pin rows use
`videoTGrid` / `writeAudioGridOffset` on the target grid starting at that
post-ref cursor. Row indices are disjoint (`row` only increases); asserts at
the end tie `seq_len` to the sum.

RoPE: ref blocks write their own `(t,h,w)` via `RefGrid` / `writeAudioGrid`;
pins sit on the target `t_grid` slice. No shared row gets two segment kinds;
`img_update`/`audio_update` mark refs as non-denoised cond rows and pins as
denoise targets (same as today separately).

AdaLN: `buildTimestepPlan` already treats `.ref_img` / `.video_pin` as visual
cond class and `.ref_audio` / `.audio_pin` as audio cond class (timestep fix
`b9e6e82f`). No mutex in the plan builder.

**Audio refs + audio pins:** rows do not overlap, but both ride the audio
stream with separate time coordinates. Revisit if voice+pin sounds wrong in the
GPU probe; layout does not force a separate refusal.

## Open questions (answer in step 1 or the probe)
- **Audio:** does a voice ref fight a pinned audio prefix? If so, allow video
  refs with pins first and keep audio refs + audio pins refused by name.
- **Pull strength:** with a strong ref, does each window snap toward the
  ref's pose or framing? That would be the same looping risk as the capstone,
  just weaker.

## GPU probe (the user's explicit go per render; h3-timeline runs it)
The same walk plan as h3-timeline `.pi-work/advisor/gpu/long/walk-none/plan.json`
(60 s, 512², 6 steps, turbo LoRA v4, seed 2207), with refs on every segment.
Two ref sources, one render each:
- **(a) self-reference:** segment 1's first frame as an image ref. It needs
  no Cast setup and anchors identity without pinning a frame position.
- **(b) a Cast image** of the man, if the user supplies one.

Compare against walk-none and walk-keystone at 20/40/55 s for face, hair and
sharpness, and check for no looping. Memory is tight: 512² × 158 frames
already needs ~26.9 GB, so refs add rows.

## Downstream (h3-timeline; a separate goal, the user's decision)
D-80/D-82 limit refs to segment 1 of a Long take, and resume refuses a changed
ref selection. Once `ref_pins` ships, h3-timeline would:
- gate on it;
- send the take's refs on every segment;
- offer the "self-reference" option.

All of that needs a design pass and a DECISIONS amendment first.
