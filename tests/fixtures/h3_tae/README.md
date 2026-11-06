# H3 TAE Core ML fixtures

Optional latent sidecar for bridge/verify scripts (not checked in — ~100 KB fp16):

1. Export from take `0683A2D2340B` (704×704, T=22, H=W=44, C=24):

   ```bash
   python3 /tmp/h3-tae-probe/export_latent_bin.py   # or copy from h3-timeline probe
   ```

2. Set env for tests:

   - `H3_TAE_LATENT_BIN` — `video_latent_fp16.bin` (NTHWC fp16 on disk, loader converts to f32)
   - `H3_TAE_LATENT_META` — JSON with `t`, `h`, `w`, `c`

Default paths: `/tmp/h3-tae-probe/latent_export/`.
