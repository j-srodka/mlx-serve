#ifndef MLXSERVE_H3_TAE_PREVIEW_H
#define MLXSERVE_H3_TAE_PREVIEW_H

#include <stddef.h>
#include <stdint.h>

/* Optional Core ML tiny decoder for MiniMax H3 live previews (MIT taeh3 weights,
 * shipped as a .mlmodelc bundle). Runs on ANE when use_ane != 0. Never required
 * for generation — linear latent_rgb is the fallback. */

typedef struct h3_tae_ctx h3_tae_ctx;

int h3_tae_preview_available(void);

/* bundle_dir = path to one *.mlmodelc directory (e.g. …/768.mlmodelc). */
h3_tae_ctx *h3_tae_preview_open(const char *bundle_dir, int use_ane, char *error,
                                size_t error_size);
void h3_tae_preview_close(h3_tae_ctx *ctx);

/* Decode one latent-frame preview. `latent` is f32 NHWC [T,H,W,C] row-major.
 * `t_idx` selects which latent time slice to show (uses a 2-frame chunk with
 * zero state, matching Draw Things / taehv streaming decode). Writes RGB888 into
 * `out_rgb` (caller allocates H*16*W*16*3 bytes). */
int h3_tae_preview_decode_frame(h3_tae_ctx *ctx, const float *latent, int t, int h, int w,
                                int c, int t_idx, uint8_t *out_rgb, int *out_pw, int *out_ph,
                                char *error, size_t error_size);

#endif
