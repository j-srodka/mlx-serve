**Status: RESOLVED 2026-10-28** — Phase A shipped: `latent_out` with refs on hybrid packs (`refCompatError`, sidecar excludes reference rows, `GET /health` `"ref_latent_out":true`). Phase B (pins with refs) remains open.

# Handoff — mlx-serve: let REF2VA references coexist with latent sidecars (`latent_out`, then pins)

Created 2026-10-26 from the h3-timeline priority grill. Repo for this work:
`~/Github/mlx-serve` (branch `local/ornith-h3` @ `305e53d3`, "merge upstream
v26.10.3"; `lib/mlxc-src` was already modified). Consumer repo:
`~/Github/h3-timeline` (see its `docs/LATER.md` § Priority item 4, PARKED). Check
both for drift before acting. Precedent for this kind of guard lift:
`handoff-keyframe-latent-lift.md` (RESOLVED 2026-10-15: keyframes × sidecars).

## Why we want it

The user runs a **hybrid FL2VA+REF2VA pack** on 8080
(`start-minimax-h3-hybrid.sh`, config tasks `t2va, fl2va, ref2va`), so model
residency is no longer the blocker. The H3 Timeline app builds everything that
matters (Extend, Prepend, Bridge, Loop, Long takes) from **latents**: a take's
`latent_out` sidecar becomes the next request's `latent_prefix_path` /
`latent_suffix_path` pin.

Today the identity workaround is a First-frame still. A **RefMod pack**
(identity / body / voice `.safetensors` via `ref_mod_paths`; skill
`minimax-h3-refmod`) holds identity far better, but a request carrying refs
gets **no latents**, so a refmod take is a cut-only dead end: it can never be
extended or bridged. Lifting the guard lets a film open on a refmod take and
continue with ordinary FL2VA joins.

## Evidence (verified 2026-10-26, read-only)

- `src/gen.zig` ~4572–4581: `latent_prefix_frames` / `latent_suffix_frames`
  with refs → 400; then
  `if ((has_latent_out or has_latent_prefix_path or has_latent_suffix_path) and (has_live_refs or has_ref_mod_fields or refs.items.len > 0))`
  → 400 "latent sidecar fields cannot be used with references or ref_mod_paths".
  The comment just above says `latent_out` "stays orthogonal — it writes the
  finished run's latents after generation returns, nothing to do with
  conditioning" (the same argument that made it legal with keyframes).
- `src/gen.zig` ~4791: `latent_out` writes `res.video_latent` / `res.audio_latent`
  via `minimax_h3.saveLatentSidecar`.
- `src/minimax_h3.zig` ~640–676: reference rows (`ref_img_rows`, `ref_audio_rows`)
  are separate CONDITION segments of the sequence
  (`text + cond + ref_img + ref_audio + audio + video`). `~5395`:
  `zlat = unpatchifyVideoRows(video_x, shape.latent_t, …)` and `~5412`
  `audioRowsToLatent(audio_x, shape.audio_t, …)`. **Hypothesis to verify:**
  `video_x` / `audio_x` hold only the generated rows, so the saved latent already
  excludes the reference rows and `latent_out` is safe as-is.

## Requirements (from the H3 Timeline side)

**Phase A — must (unblocks the app):**
1. `latent_out` accepted with `ref_mod_paths` and with live refs
   (`<Picture N>`, `<Video N>`, `<Audio N>`). The written sidecar has the SAME
   format and shape as a no-ref FL2VA run of the same canvas/frames: generated
   video + audio latents only, **no reference rows** (a voice pack's audio rows
   especially must not leak into `audio_latent`).
2. A sidecar written under refs works unchanged as a later NO-ref request's
   `latent_prefix_path` / `latent_suffix_path` (the app's Extend / Bridge from a
   refmod take). Test: refmod take → `latent_out` → plain FL2VA extend pinned on
   it, seam as clean as an all-FL2VA extend.
3. Still a hybrid/FL2VA-only feature: on a pure REF2VA pack keep the named 400
   (`latent_out is FL2VA only`).
4. Hermetic test for the shape/row-exclusion (like `keyframeSidecarBindingError`)
   plus the named-400 matrix updated.
5. Advertise it so the app can gate without trial-and-error: e.g. `GET /health`
   gains `"ref_latent_out": true` (name up to you; the app currently reads
   `latent_suffix`).

**Phase B — should, separate decision (bigger):**
6. Pins (`latent_prefix_path` / `latent_suffix_path`, and the frame counts) WITH
   refs in the same request, so every window of a Long take or a Bridge keeps the
   identity pack, not just the opener. Open questions: RoPE time-grid placement of
   pinned rows next to reference rows; whether a voice pack fights a pinned audio
   prefix (the app already has audio restarts at joins — LATER "Audio
   continuation across loop-window joins"); quality probe on a real pack before
   shipping. If this is not sound, keep the per-side named 400s.

**Non-requirements:** no change to chain windows (`chain_windows > 1` stays
FL2VA-keyframe only); no change to request shapes the app already sends; the app
never starts/stops the server.

## Constraints

- Follow mlx-serve's own AGENTS.md (small diffs, class guards, sidecar
  `./local/export.sh` before/after a parent commit, docs in `docs/reference.md`
  request surface + `skills/mlx-serve/media.md`).
- GPU probes need the user's explicit go per render; memory is tight at 704²
  (2026-10-26: a 192-frame 704² window was refused "needs ~32.7 GB … 30.2 GB
  available" with apps open).

## Downstream (h3-timeline, separate follow-up)

When Phase A ships: a D-n in h3-timeline `docs/DECISIONS.md` reversing the REF2VA
non-goal for root takes; refmod pack picker on New take; record packs in
`take.json` for reroll parity; gate on the new `/health` flag.

## Suggested skills

`minimax-h3-refmod` (pack facts), `minimax-h3-timeline` (consumer contract),
`grilling` / `grill-me` before Phase B.
