# Handoff: mlx-serve, Late Lock and conditioning-only pins for H3 FL2VA

Created 2026-11-05 from the h3-timeline advisor session, at the user's request.
Repo for this work: `~/Github/mlx-serve`.

**Drift check before acting:**
- mlx-serve: HEAD `4ed23688` on branch `cursor/h3-tae-ane-preview-32b0`, with
  uncommitted changes the user owns (`docs/reference.md`,
  `handoff-h3-tae-preview.md`, `lib/h3_tae/h3_tae_preview.m`, `lib/mlxc-src`,
  `skills/mlx-serve/media.md`). Don't touch them; branch per this repo's push
  topology (`docs/gotchas/repo-topology.md`).
- Consumer repo: h3-timeline main @ `bbd5528`.

## Summary

The h3-timeline app's "beats" (a visible ~1.6 s return to a long take's opening
look) snap visibly in every variant measured on 2026-11-05.

The original idea comes from roycho87/clown_context_suite (now the "Geiru
Context Suite"). It pulls each shot back toward clip 1 **invisibly**:
- **Keystone:** clip 1's frame-0 latent sits in latent 0, and the real frame is
  restored after.
- **Capstone:** clip 1's end latent is tacked after the shot and cut off.
- **Late Lock:** anchors come in only for the last ~25 % of steps.
- **Context noise:** held frames get partial noise that fades to none at the
  join.

h3-timeline can do keystone and capstone **client-side today**, by writing the
pin files itself and trimming. It's building that as a CLI-only experiment
("hidden anchor"). **Late Lock and context noise need this server.** This
handoff specifies them.

Source read, feasibility, and the source's numbers (drift 6.4 → 3.9 keystone →
2.9 with a late-locked capstone, over 10 clips): see
`/Users/jsrodka/Github/h3-timeline/.pi-work/advisor/beats-source-review.md` and
`/Users/jsrodka/Github/h3-timeline/.pi-work/advisor/hidden-anchor-feasibility.md`.

**Status:** plan only. No code written. **Gate:** start only after the
client-side hidden-anchor sitting shows a real drift reduction. The user decides.

## How pins work today (verified 2026-11-05, read-only)

- **Validation, `src/gen.zig:4418-4454`:** prefix and suffix may coexist with
  `chain_windows == 1`. Sizes go through `minimax_h3.zig`
  `latentPrefixRequestError` / `latentSuffixRequestError` (5601-5636): sizes in
  {5, 22, 39, 56}, each less than the window. `latentPinWindowError`
  (5680-5700) requires ≥ 1 free video latent frame and ≥ 1 audio step.
- **Layout, `PackedLayout.initFull`, `minimax_h3.zig:793-860`:** pins are
  `.video_pin` / `.audio_pin` segments at the head and tail of the window's
  time grid.
- **Pin rows' timestep, `minimax_h3.zig:~2083-2088`:** they get
  `@max(t_v, aug.visual)`, i.e. "conditioning". `.video` rows get `t_v`.
- **Setup, `minimax_h3.zig:5011-5101`:** the prefix takes the **tail** of the
  prefix file's latent (`sliceLatentTRange(pv, pv_t - k, pv_t)`). The suffix
  takes the **head** of the suffix file's latent (`sliceLatentTRange(sv, 0, k)`).
  They're spliced into `video_x` / `audio_x` via `prependPinnedRows`,
  `appendPinnedRows` and `spliceAudioPinLayout` (4262-4316).
- **Every step, `minimax_h3.zig:5288` loop:** both the step-cache skip path
  (~5326) and the full-step path (~5418) call `restorePinnedLatent(...)`, which
  overwrites the pinned rows with the clean pin after each update.
- **Output:** the output head concatenates `.video_pin` + `.video`
  (`concatTargetStreamHead`, ~3002), so pinned frames are in the decoded clip
  and in `latent_out`.
- **Health, `src/server.zig:2623`:** emits `latent_suffix` / `ref_latent_out`.

## The changes (priority order)

### 1. Late Lock: `latent_prefix_from_step` / `latent_suffix_from_step` (u32, default 0)
- **Parse** in `gen.zig` next to `latent_*_frames`, and add the fields to the
  request struct (`minimax_h3.zig:~3970-3976`). Validate in a new pure fn
  `latentPinStepError(from_step, steps, frames)`: refuse a non-zero value
  without its pin, and refuse `from_step >= steps`, each with a named 400.
- **Before `from_step`, the pin is inactive:**
  - its rows start as the seed noise (skip `prependPinnedRows` /
    `appendPinnedRows` for that side at setup);
  - `restorePinnedLatent` skips that side;
  - the AdaLN run table treats that side's `.video_pin` / `.audio_pin` segment
    as `.video` / `.audio` (timestep `t_v` / `t_a`).

  Simplest route: thread a `pins_active: struct { v_prefix, v_suffix, a_prefix,
  a_suffix: bool }` into the per-step run builder, rather than rebuilding
  `PackedLayout`. Position ids don't change.
- **At step `from_step`:** overwrite that side's rows with the clean pin, then
  behave exactly as today.
- **Step cache:** a step whose pin activation changes must not be a skipped
  step. Force `sc_skip = false` on the swap step, and reset `sc_acc`.
- **Tests (hermetic):**
  - `from_step` 0 is byte-identical to today (same seed and layout, compare
    `video_x` after step 1);
  - before the swap, the pin rows equal the seed noise path, not the pin;
  - from the swap on, the rows equal the pin;
  - the named 400s.

### 2. Conditioning-only suffix: `latent_suffix_output: false` (default true)
- The suffix pin still joins the sequence and conditions every step, but its
  rows are dropped from the decoded video and from `latent_out` (video and
  audio).
- **Frame math:** today `num_frames` includes the pin. With output off, decode
  `num_frames − suffix_frames` (or document that the client trims). Pick
  whichever keeps the 17k+5 ladder legal for the **output**. Recommendation:
  keep `num_frames` as the total window and drop the suffix's latent rows
  before decode. The VAE's causal first row is untouched, since it's a tail
  drop.
- **Audio:** drop the matching audio latent steps (`overlapAudioLatentT`) and
  their PCM samples (`chainOverlapPcmSamples` pattern).
- **Tests:** output latent T = `latent_t − pin_v_suffix_t`; `latent_out` shapes;
  default true is byte-identical to today.

### 3. Context noise: `latent_pin_noise: f32` in [0, 1] (default 0)
- At each step, while a pin is active, the restored pin rows get
  `pin + noise × σ_current × ramp(row)` instead of the clean pin. `ramp` is 1 at
  the pin's outer edge and falls linearly to 0 at its inner edge (the join).
  The noise is the sampler's current noise for that row, kept from the seed
  stream so runs stay deterministic.
- The timestep stays "conditioning" (the source does the same: the model is told
  the rows are held, but they aren't exact).
- **Tests:** 0 is byte-identical; a value > 0 changes only the pin rows; the
  ramp's endpoints; determinism across two runs.

### 4. Capability flags
`/health` adds `"late_pin": true`, `"pin_output": true`, `"pin_noise": true`
(`server.zig:2623` builds the body: extend the format). h3-timeline gates on
these like `latent_suffix`.

### 5. Docs in this repo
- **`docs/reference.md`** (H3 request surface): the three fields, their
  defaults, their 400s.
- **`skills/mlx-serve/media.md`:** one line per field.
- **AGENTS.md:** a rule only if a real gotcha turns up (growth policy).

## GPU acceptance (in this repo, one render at a time, the user's go each)
- **Plan:** the h3-timeline studio plan,
  `/Users/jsrodka/Github/h3-timeline/docs/studio-current.json`. Its prompt,
  seed 1107, 512², 15 steps, no LoRA; render through h3cli once the client
  passes the fields.
- **Arms:**
  1. `from_step` 0 vs ~75 % of steps (11 of 15), with a capstone suffix 5;
  2. then `latent_pin_noise` 0 vs 0.5.
- **Measure:** `h3cli drift <take>` (D-102 limits). The baseline is the
  no-beats studio take rendered 2026-11-05 under `~/Movies/H3-Studio-No-Beats`.

## Action items (priority-ranked)
1. Confirm the gate: the user has a client-side hidden-anchor result that's
   worth extending.
2. Implement change 1 (Late Lock) with its tests, then build and run the H3
   unit tests.
3. Implement change 2, then change 3, each with byte-identical-default tests.
4. Add the health flags and docs.
5. Hand back to h3-timeline: its consumer work (request fields, gating, plan
   keys) is a separate goal there, written by its advisor.

## What to do next
Read `minimax_h3.zig:5011-5101` and the step loop at 5288-5440, then write the
`pins_active` threading for change 1 behind a byte-identical test.

## Suggested skills
- `mlx-serve` (this repo's own skill, `skills/mlx-serve/`) for the media API
  surface.
- `adversarial-review` before calling a change done.
- `grill-me` if the user hasn't settled the frame-math choice in change 2.
- `h3-timeline` (`~/.agents/skills/h3-timeline`) only for rendering the GPU
  acceptance plans through h3cli.
