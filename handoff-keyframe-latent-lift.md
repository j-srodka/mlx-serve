# Handoff — mlx-serve: lifting the keyframe × latent-sidecar guardrail (FL2VA)

> **RESOLVED 2026-10-15** (commits `f1a85600`, `0a90a51a` after the v26.10.2 merge): A =
> allowed outright (code-verified: `latent_out` writes post-generation, orthogonal to
> conditioning). B = per-side named 400s (`.first`+prefix pin and `.last`+suffix pin
> refused; mirror combos legal; pure fn `keyframeSidecarBindingError` + hermetic tests).
> C = chains INCLUDED, head AND tail file seeding (intentional scope expansion over the
> original C recommendation below, per interview): window 0 opens on the prefix file's
> tail; the final window finishes at the suffix file's head; `var wreq = req` suffix
> fields are cleared on windows 0..n-2 (cursor-review findings #1/#2/#4/#9 folded in;
> delivered-frame math unchanged — measured post-join). Docs: `docs/reference.md` request
> surface + chained windows, `skills/mlx-serve/media.md`. Downstream h3-timeline unlocks
> still parked in its docs/LATER.md.

Created 2026-10-15 from the h3-timeline WP6.6 session. Repo for this work:
`~/Github/mlx-serve`. Drift check: `git -C ~/Github/mlx-serve log -1` — known
HEAD at handoff: `7aeec2bd` ("chore(git): enforce push topology…"; `lib/mlxc-src`
was modified). The consumer repo's HEAD was `/Users/jsrodka/Github/h3-timeline`
@ `4df0ffb` (WP6.6 shipped); check both for movement before acting.

## Goal

Let `POST /v1/video/generations` accept **FL2VA keyframe anchors
(`first_frame_image` / `last_frame_image`)** together with the latent-sidecar
fields it currently refuses: `latent_out`, `latent_prefix_path`,
`latent_suffix_path`. Today it returns ONE lumped 400 (see evidence). The
h3-timeline app then unlocks (separate follow-up, tracked there — see
"Downstream unlocks").

## Evidence already gathered (do NOT re-derive; verified 2026-10-15)

1. **The refusals are in one check** — `src/gen.zig` line ~4098:
   ```zig
   if ((has_latent_out or has_latent_prefix_path or has_latent_suffix_path) and
       (has_live_refs or has_ref_mod_fields or refs.items.len > 0 or n_kf > 0))
       return sendError(conn, 400, "latent sidecar fields cannot be used with references, ref_mod_paths, or keyframes");
   ```
   The **refs** part is genuinely incompatible (`src/minimax_h3.zig`
   `generateChain` returns `error.ChainWithRefs`); only the **keyframes**
   (`n_kf > 0`) part is the conservative piece.
2. **The engine already mixes anchors with latent pins.** `generateChain`
   (`src/minimax_h3.zig` ~5314): window 0 takes the user's `.first` keyframe;
   the **final window takes the user's `.last` keyframe *together with*
   `chain_prefix_video_latent != null`** (`kfs` + `wreq.chain_prefix_video_latent`
   are set on the same `wreq` passed to `generateOne`). So
   anchors + latent prefix/pins is proven workable at inference level.
3. **Live-verified matrix** (real server, real GPU, 2026-10-15; reuse
   `/tmp/wp6.6/probe_matrix.py`-style bodies if still present, else §Matrix
   below):
   - `first_frame_image` alone (22 f, and even 5 f) → ran, returned video.
   - `first + last` (22 f) → accepted (slower: two-stage pipeline).
   - `last` alone → accepted.
   - keyframes + `latent_out` → **400** (the lumped message).
   - keyframes + `latent_prefix_path` / `latent_suffix_path` → **400**.
   - **`chain_windows:2` + `latent_prefix_frames:22` + `first_frame_image`
     (no paths, no `latent_out`) → accepted; generation started** (SSE
     "Window 1/2: Encoding prompt").
   - malformed base64 → **400 "keyframe image is not valid base64"** (named,
     keep).
   - `num_frames:1` stills reject keyframes by name (keep).
   - Note: the "keyframes need at least 9 frames" (`KeyframeCanvasTooShort`)
     path did NOT fire at 5 frames — an accepted 5-frame conditioned run was
     observed; don't re-add a 9-frame floor speculatively.
4. **Handler geometry rules to preserve** (`src/gen.zig` ~4020–4065):
   `first` anchor = plain stretch to the generation canvas (the geometry
   anchor); `last` = aspect-preserving center-cover. The keyframe also enters
   the Qwen conditioning as a `<Picture i>` block (separate canvas rules, a
   second decode only when canvases diverge). Bare base64 in, no data-URI
   prefix; PNG/JPEG only.

## The open design questions (this is the actual investigation)

A. **`latent_out` + keyframes — why was it lumped in, and is it safe?**
   Hypothesis: over-broad grouping with refs. The output latent is just the
   run's final latent; writing it should be orthogonal to conditioning.
   Verify by reading the latent_out write path end-to-end
   (`src/gen.zig` ~4100–4200 validation + wherever `latent_out` is written
   in `minimax_h3.zig`) and the sidecar pack validation
   ("sidecar must declare pack fl2va", ~4171+). Watch: nothing special-cases
   conditioned runs downstream.
   **Risk the consumer needs checked**: an anchor-influenced latent, later
   reused as a pin, is semantically "the continuation of the anchored
   content" — that is exactly what extend should do; no D12-style mapping
   problem exists for single-window runs. Document this reasoning in code.

B. **Pin paths + anchors in extend/prepend shapes — what binds frame 0?**
   With `latent_prefix_path`, the window's first frames are the HELD TAIL of
   the previous take (seam-continuation content). A user `.first` anchor
   would fight the pin for the same latent frame. Candidate semantics (pick
   one, implement, test):
   1. Refuse keyframe+pin *per side* with a NAMED 400 (e.g. "a first-frame
      anchor cannot combine with a prefix pin — the pin owns the window's
      opening"), keep last-frame anchor legal with prefix pins (mirror of the
      proven final-window combo), and vice versa for suffix pins. — the
      conservative-but-useful middle.
   2. Bind the user anchor to the first NEW frame (post-hold) — needs engine
      plumbing (anchor offsets by pin length); do only if the model layer
      makes it cheap.
   Use `generateChain`'s final-window behavior (`user .last` + prefix pins)
   as the precedent when deciding.

C. **Chain (>1 window) with keyframes + path sidecars** — currently latent
   pin PATHS require `chain_windows == 1` (separate 400s, ~3960–3966); the
   chain path uses in-memory prefix latents instead. Scope decision: stay
   out (keep those 400s), or include. Recommendation: stay out for v1 of the
   lift; the frame-pin chain combo (verified working, item 3) is the chain
   feature that matters.

## What to change (expected shape)

- `src/gen.zig`: split the lumped check — refs stay banned with sidecar
  fields; keyframes become compatible with `latent_out` always, and per the
  B-decision with pin paths. All other named 400s keep their behavior.
- `src/minimax_h3.zig`: only if B-decision requires engine plumbing.
- Tests: follow whatever test conventions the repo has
  (`grep -rn "test " src/ | head` — there are in-source Zig tests); add
  request-validation cases per matrix; the server has a live-GPU checklist
  culture — see `AGENTS.md` + `README.md` in the repo (READ BOTH FIRST; the
  repo contract lives there, including the push topology rule from HEAD).
- Rebuild/restart affects the RUNNING server that h3-timeline depends on —
  coordinate: check there is no in-flight generation before restart
  (the app shows "Generating…"; `curl /health` stays 200 while idle).

## Matrix (curl against the restarted server; run ALL, record verbatim)

Base body: `{"prompt":"probe","width":256,"height":256,"steps":4,"seed":7}`
(+ `stream:true` and read the FIRST SSE event, then disconnect — that alone
proves validation passed and cancels cleanly server-side). 256×256 tiny
non-stream runs are cheap (~<1 min) when you want full completions.

| probe | expect after lift |
|---|---|
| kf + `latent_out` (absolute path, existing parent e.g. `/tmp/wp6.6/out.safetensors`) | 200 |
| kf + `latent_output` sidecar contents: verify written pack declares `fl2va` + `video_latent` | assert |
| kf + `latent_prefix_path` (extend shape: prefix_frames 22, `chain_windows:1`, num_frames 39) | per A/B decision — 200 with defined binding, or NAMED 400 (not the lumped message) |
| kf + `latent_suffix_path` (prepend shape) | same as above |
| last-frame-only kf + `latent_prefix_path` | 200 (the proven engine combo's HTTP shape) |
| refs (`ref_images`) + any sidecar field | still 400 (unchanged) |
| kf + `chain_windows:2` + `latent_prefix_frames` | still 200 (unchanged) |
| malformed kf base64 / num_frames 1 + kf | still named-400 (unchanged) |
| t2v no-anchor request with `latent_out` (regression) | byte-identical behavior |

Latent sidecar pack for path probes: a real project take's
`latents.safetensors`, e.g. `/tmp/h3-wp2-demo/takes/5E582E1CB516/latents.safetensors`
(256×256, 311 frames) — /tmp is volatile; regenerate from any project take
folder if gone. Anchor images for probes: any small PNG (`ffmpeg -f lavfi -i
color=c=red:s=256x256 -frames:v 1`).

## Downstream unlocks (record in this repo's notes or the h3-timeline repo)

Once lifted, h3-timeline (in `/Users/jsrodka/Github/h3-timeline`) can, as a
small follow-up slice (currently parked in its `docs/LATER.md` bullet
"Lift the mlx-serve keyframe × latent-sidecar guardrail"):
1. conditioned takes keep `latent_out` → drop the `TakeIntegrity.conditioned`
   terminal gate (keep the case for takes generated pre-lift),
2. re-enable Source picker in seam-armed contexts (pin+anchor semantics permitting),
3. chain-window conditioning UI (the frame-pin combo already verified).

App-side compatibility is preserved DURING the lift either way: the app
today OMITS `latent_out` for conditioned runs, and the lifted server must
still accept that.

## Suggested skills for the receiving agent

- Read `~/Github/mlx-serve/AGENTS.md` + `README.md` first — that repo has its
  own contract (push topology, build/run flow) and this handoff does not
  duplicate it.
- `grill-me` / `/gather-context-and-clarify` — settle design decision B
  (pin × anchor binding) with the user before implementing.
- `adversarial-review` — review the guardrail split the same way the
  h3-timeline slices were reviewed (numbered findings, VERDICT format).
- `computer-use` — if the h3-timeline app-side unlock is verified live later
  (it drives the packaged app via accessibility).

## No secrets, tokens, or personal data are involved in this work.