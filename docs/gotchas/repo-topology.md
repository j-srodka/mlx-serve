# Repo topology, remotes and the backup workflow (oct 2026)

**Root cause of every git mixup here:** three remotes with similar names and
two lines of development whose roles are not visible in `git log`. One agent
attempted a ddalcu push; another nearly deleted the only local copy of the
H3 native-port branch whose upstream had already been deleted.

## The three remotes and what they are for

| Remote / repo | Role | Push policy |
|---|---|---|
| `origin` = ddalcu/mlx-serve | upstream we track | **NEVER push** (URL is `DISABLED_NO_PUSH`; keep it that way) |
| `fork` = j-srodka/mlx-serve | two jobs: `fork/main` mirrors where we've PULLED ddalcu upstream to (staging line, reviewed before integrating into ornith), and holds branch backups | branch pushes only (`local/ornith-h3`, `backup/*`); never update `fork/main` with custom work |
| `~/Github/mlx-serve/local` = j-srodka/mlx-serve-local (sidecar) | patch/overlay backup independent of any branch state | its own `main`, push freely |

## The two lines of development

- `local/ornith-h3` — the CUSTOM line (H3, refmods, chain windows, LTX Ripple,
  timeline tooling). Custom work **is committed in the parent here** (the
  sidecar README's older "don't commit in parent" rule is superseded), and
  pushed to `fork/local/ornith-h3`.
- `fork/main` — 176+ commits ahead of ornith with other active work (CUDA,
  app features). It is NOT behind us; don't rebase ornith onto it
  opportunistically. Reconciliation is a deliberate reviewed session:
  integrate, rebuild, run `tests/` + H3 Timeline against it.

Local `main` is a throwaway pointer that was fast-forwarded to `fork/main`
(oct 2026) so its name matches its ddalcu-staging role.

## Sidecar workflow (j-srodka/mlx-serve-local)

1. `./local/export.sh` **BEFORE** the parent commit — captures the dirty
   tracked tree as one patch per file + copies untracked keepables into
   `overlay/`, records `BASE` (= parent HEAD).
2. Commit + push in the sidecar.
3. After the parent commit lands, re-run `export.sh` once more: patches/
   empty, `BASE` bumps to the new commit. Skipping this leaves patches that
   represent work already committed — a later `apply.sh` after an upstream
   fast-forward would try to double-apply them.
4. `./local/apply.sh` restores the sidecar onto a freshly fast-forwarded
   parent (git apply --3way). Never `git clean -fd` without a pathspec —
   it deletes `src/h3_refmod.zig` and friends.

## One-time traps, pinned

- **Untracked import = broken branch.** `f8a4feed` committed `gen.zig`'s
  `@import("h3_refmod.zig")` but NOT the file — a clean clone of that state
  does not build. When agent sessions commit engine changes, list the new
  files explicitly; `git status` untracked is your checklist.
- **Dirty submodule is by design.** `lib/mlxc-src` shows `mlx/c/ops.cpp`
  modified: `scripts/build-mlx.sh` applies `patches/mlxc-gather-qmm-*.patch`
  onto it at build time. Don't stage the gitlink, don't "restore" it, never
  commit inside the submodule.
- **Self-matching greps in test scripts.** A skip-conditional written as
  `grep -q 'flag' "$0"` always matches once the script mentions the flag in
  its echo — the H3 media peak check silently disabled itself this way
  (fixed: `SKIP_MEM_PREFLIGHT=1` env toggle).
- **H3 Timeline depends on this tree**, not GitHub: `~/Github/h3-timeline`
  calls `/v1/video/generations` and launches `start-minimax-h3-hybrid.sh`
  (launcher default port 11361; its client and `h3_timeline.py` default 8080 —
  set `MLX_SERVE_PORT=8080` or pass `--url` explicitly).
- The pre-commit backup branches `backup/minmax-h3-native-port` and
  `backup/ref2va-polish` on `fork` hold the pre-ornith custom lines (18
  commits combined). The local `feature/*` branches were deleted
  (oct 2026) once those landed; old stash `h3-wip-before-v26.9.1` kept
  until H3 Timeline verifies against the new build.