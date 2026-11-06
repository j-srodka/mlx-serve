# Handoff — mlx-serve: number audio RefMods and live audio refs on ONE counter

Created 2026-10-28 from the h3-timeline Cast (REF2VA) plan, decision D-60 in
`~/Github/h3-timeline/docs/DECISIONS.md`. Repo: `~/Github/mlx-serve` (branch
`local/ornith-h3`; the working tree has unrelated uncommitted changes — leave them).
Small, self-contained; no GPU needed to prove the label text (hermetic test).

## Problem (verified in source 2026-10-28)

H3 reference labels are 1-based PER TYPE (`src/minimax_h3_vision.zig` `labelFor`:
`<Picture n>`, `<Video n>`, `<Audio n>`). Two places count audio independently:

- `src/gen.zig` `parseH3RefMods` (~3900-3934): `n_aud` from 0 for `ref_mod_paths` of kind
  audio (capped at `MAX_REF_AUDIOS`).
- `src/minimax_h3.zig` `resolveRefs` (~352-406): `n_img/n_vid/n_aud` from 0 again for the
  live `ref_images` / `ref_videos` (soundtracks) / `ref_audios`.
- The label loop (`src/minimax_h3.zig` ~4821-4830) emits `<Audio j>` for an audio RefMod
  (latent present, kind audio) using ITS ordinal, then the live refs with theirs.

So an audio RefMod + a live `ref_audios` entry (or a soundtracked `ref_videos` entry,
whose soundtrack is an `<Audio j>`) emits **two `<Audio 1>` labels**. The handler allows
this mix on purpose (gen.zig ~4552: "Packs + live video/audio is the character+scene /
dual-voice mix"), so the prompt cannot address the two voices separately.

Image/video RefMods emit no label (identity rows), so only audio collides.

## Ask

Number audio refs on ONE counter across RefMods and live refs, in the order the
rows are presented (RefMods first per gen.zig ~4560, then live refs: video soundtracks
before their `<Video k>`, then standalone audio). E.g. pass the RefMod audio count
into `resolveRefs` as its starting `n_aud`. Also apply the total audio cap
(`MAX_REF_AUDIOS` = 3) across both, or document why not. Update docs/reference.md ~205
(ordinals) and add a hermetic test: 1 audio RefMod + 1 `ref_audios` →
labels `<Audio 1>` (pack) and `<Audio 2>` (live).

## Consumer side

Until this lands, H3 Timeline refuses (by name, before POST) any take that mixes an
audio RefMod with live ref audio or a soundtracked ref video. When it lands, tell the
h3-timeline side (docs/LATER.md item) so the app can lift that refusal and number the
chips to match.

