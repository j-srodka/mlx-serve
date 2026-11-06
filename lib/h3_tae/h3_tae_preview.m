#import "h3_tae_preview.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef __fp16 f16;

struct h3_tae_ctx {
    MLModel *model;
    int chunk_t;
    int pad_h;
    int pad_w;
    int latent_c;
};

static void fail(char *error, size_t error_size, const char *fmt, ...) {
    if (!error || !error_size) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

int h3_tae_preview_available(void) {
#if TARGET_OS_OSX
    return NSClassFromString(@"MLModel") != nil;
#else
    return 0;
#endif
}

static void zero_f16(MLMultiArray *arr) {
    if (arr.dataType != MLMultiArrayDataTypeFloat16) return;
    f16 *p = arr.dataPointer;
    for (NSInteger i = 0; i < arr.count; i++) p[i] = (f16)0;
}

static f16 f16_at(const MLMultiArray *a, const int idx[5]) {
    const f16 *base = a.dataPointer;
    size_t off = 0;
    for (NSInteger d = 0; d < a.strides.count; d++) {
        off += (size_t)idx[d] * (size_t)a.strides[d].integerValue;
    }
    return base[off];
}

static void log_array_strides(const MLMultiArray *a, const char *name) {
    if (!getenv("MINIMAX_H3_TAE_DEBUG")) return;
    NSMutableString *sh = [NSMutableString stringWithFormat:@"[h3-tae] %s shape=", name];
    NSMutableString *st = [NSMutableString stringWithString:@" strides="];
    for (NSInteger i = 0; i < a.shape.count; i++) {
        [sh appendFormat:@"%ld", (long)a.shape[i].integerValue];
        [st appendFormat:@"%ld", (long)a.strides[i].integerValue];
        if (i + 1 < a.shape.count) {
            [sh appendString:@","];
            [st appendString:@","];
        }
    }
    fprintf(stderr, "%s%s\n", sh.UTF8String, st.UTF8String);
}

static MLMultiArray *make_chunk(const float *latent, int T, int H, int W, int C, int chunk_t, int pad_h,
                                int pad_w, int source_start, int real_frames, NSError **err) {
    NSArray<NSNumber *> *shape = @[ @1, @(chunk_t), @(pad_h), @(pad_w), @(C) ];
    MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:err];
    if (!arr) return nil;
    f16 *dst = arr.dataPointer;
    for (NSInteger i = 0; i < arr.count; i++) dst[i] = (f16)0;
    for (int f = 0; f < real_frames; f++) {
        const int src_t = source_start + f;
        if (src_t < 0 || src_t >= T) continue;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                const size_t src_base = (size_t)(((src_t * H + y) * W + x) * C);
                const size_t dst_base = (size_t)(((f * pad_h + y) * pad_w + x) * C);
                for (int ch = 0; ch < C; ch++) {
                    dst[dst_base + (size_t)ch] = (f16)latent[src_base + (size_t)ch];
                }
            }
        }
    }
    return arr;
}

static NSMutableArray<MLMultiArray *> *zero_states(MLModel *model, NSError **err) {
    NSMutableArray<MLMultiArray *> *states = [NSMutableArray arrayWithCapacity:9];
    for (int i = 0; i <= 8; i++) {
        NSString *name = [NSString stringWithFormat:@"act_%d", i];
        MLFeatureDescription *desc = model.modelDescription.inputDescriptionsByName[name];
        MLMultiArrayConstraint *c = desc.multiArrayConstraint;
        if (!c) {
            if (err) *err = [NSError errorWithDomain:@"h3_tae" code:2 userInfo:@{NSLocalizedDescriptionKey : name}];
            return nil;
        }
        MLMultiArray *s = [[MLMultiArray alloc] initWithShape:c.shape dataType:MLMultiArrayDataTypeFloat16 error:err];
        if (!s) return nil;
        zero_f16(s);
        [states addObject:s];
    }
    return states;
}

static MLMultiArray *predict_chunk(MLModel *model, MLMultiArray *chunk, NSMutableArray<MLMultiArray *> *states,
                                   NSError **err) {
    NSMutableDictionary<NSString *, MLFeatureValue *> *feats = [NSMutableDictionary dictionary];
    feats[@"latent"] = [MLFeatureValue featureValueWithMultiArray:chunk];
    for (int i = 0; i <= 8; i++) {
        NSString *in_name = [NSString stringWithFormat:@"act_%d", i];
        feats[in_name] = [MLFeatureValue featureValueWithMultiArray:states[i]];
    }
    id<MLFeatureProvider> provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:feats error:err];
    if (!provider) return nil;
    id<MLFeatureProvider> out = [model predictionFromFeatures:provider error:err];
    if (!out) return nil;
    MLMultiArray *image = [[out featureValueForName:@"image"] multiArrayValue];
    if (!image) {
        if (err) *err = [NSError errorWithDomain:@"h3_tae" code:3 userInfo:nil];
        return nil;
    }
    for (int i = 0; i <= 8; i++) {
        NSString *out_name = [NSString stringWithFormat:@"act_%d_out", i];
        MLMultiArray *next = [[out featureValueForName:out_name] multiArrayValue];
        if (!next) {
            if (err) *err = [NSError errorWithDomain:@"h3_tae" code:4 userInfo:nil];
            return nil;
        }
        states[i] = next;
    }
    // If preview ever carries act_* across chunks, densify strided `next` before re-feeding.
    return image;
}

static int is_regular_file(const char *p) {
    struct stat st;
    if (stat(p, &st) != 0) return 0;
    return S_ISREG(st.st_mode);
}

static int dest_is_shared_file(const char *p) {
    struct stat st;
    if (lstat(p, &st) != 0) return 0;
    if (S_ISLNK(st.st_mode)) return 0;
    return S_ISREG(st.st_mode);
}

static int copy_file(const char *src, const char *dst) {
    FILE *in = fopen(src, "rb");
    if (!in) return -1;
    FILE *out = fopen(dst, "wb");
    if (!out) {
        fclose(in);
        return -1;
    }
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, in)) > 0) {
        if (fwrite(buf, 1, n, out) != n) {
            fclose(in);
            fclose(out);
            return -1;
        }
    }
    int bad = ferror(in);
    fclose(in);
    if (fclose(out) != 0) return -1;
    return bad ? -1 : 0;
}

static int link_weight_bin_if_needed(const char *mlmodelc_path) {
    char bundle_root[4096];
    char src[4096];
    char dst[4096];
    char weights_dir[4096];
    strncpy(bundle_root, mlmodelc_path, sizeof(bundle_root) - 1);
    bundle_root[sizeof(bundle_root) - 1] = 0;
    char *last = strrchr(bundle_root, '/');
    if (!last) return -1;
    *last = 0;
    snprintf(src, sizeof(src), "%s/weight.bin", bundle_root);
    snprintf(weights_dir, sizeof(weights_dir), "%s/weights", mlmodelc_path);
    snprintf(dst, sizeof(dst), "%s/weight.bin", weights_dir);
    if (!is_regular_file(src)) return -1;
    if (dest_is_shared_file(dst)) return 0;
    (void)mkdir(weights_dir, 0755);
    // Import trees leave a dangling symlink to a temp probe; stat() follows it
    // so a later link() hits EEXIST. Unlink any non-file dest, then hardlink.
    (void)unlink(dst);
    if (link(src, dst) == 0) return 0;
    if (copy_file(src, dst) == 0) return 0;
    return -1;
}

h3_tae_ctx *h3_tae_preview_open(const char *bundle_dir, int use_ane, char *error, size_t error_size) {
    if (!h3_tae_preview_available()) {
        fail(error, error_size, "Core ML unavailable");
        return NULL;
    }
    @autoreleasepool {
        NSError *err = nil;
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = use_ane ? MLComputeUnitsCPUAndNeuralEngine : MLComputeUnitsAll;
        NSURL *url = [NSURL fileURLWithPath:@(bundle_dir)];
        if (link_weight_bin_if_needed(bundle_dir) != 0) {
            fail(error, error_size, "weight.bin missing or unlinkable under %s (errno=%d)", bundle_dir,
                 errno);
            return NULL;
        }
        MLModel *model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!model) {
            fail(error, error_size, "load model: %s", err.localizedDescription.UTF8String ?: "unknown");
            return NULL;
        }
        MLFeatureDescription *latent_desc = model.modelDescription.inputDescriptionsByName[@"latent"];
        MLMultiArrayConstraint *lc = latent_desc.multiArrayConstraint;
        if (!lc || lc.shape.count != 5) {
            fail(error, error_size, "bad latent input shape");
            return NULL;
        }
        h3_tae_ctx *ctx = calloc(1, sizeof(h3_tae_ctx));
        if (!ctx) {
            fail(error, error_size, "oom");
            return NULL;
        }
        ctx->model = model;
        ctx->chunk_t = lc.shape[1].intValue;
        ctx->pad_h = lc.shape[2].intValue;
        ctx->pad_w = lc.shape[3].intValue;
        ctx->latent_c = lc.shape[4].intValue;
        return ctx;
    }
}

void h3_tae_preview_close(h3_tae_ctx *ctx) {
    if (!ctx) return;
    ctx->model = nil;
    free(ctx);
}

int h3_tae_preview_decode_frame(h3_tae_ctx *ctx, const float *latent, int t, int h, int w, int c, int t_idx,
                                uint8_t *out_rgb, int *out_pw, int *out_ph, char *error, size_t error_size) {
    if (!ctx || !latent || !out_rgb || !out_pw || !out_ph) {
        fail(error, error_size, "bad args");
        return -1;
    }
    if (t <= 0 || h <= 0 || w <= 0 || c <= 0 || t_idx < 0 || t_idx >= t) {
        fail(error, error_size, "bad shape");
        return -1;
    }
    @autoreleasepool {
        NSError *err = nil;
        NSMutableArray<MLMultiArray *> *states = zero_states(ctx->model, &err);
        if (!states) {
            fail(error, error_size, "states: %s", err.localizedDescription.UTF8String ?: "?");
            return -1;
        }
        const int chunk_start = (t_idx / ctx->chunk_t) * ctx->chunk_t;
        const int real = (chunk_start + ctx->chunk_t <= t) ? ctx->chunk_t : (t - chunk_start);
        MLMultiArray *chunk =
            make_chunk(latent, t, h, w, c, ctx->chunk_t, ctx->pad_h, ctx->pad_w, chunk_start, real, &err);
        if (!chunk) {
            fail(error, error_size, "chunk: %s", err.localizedDescription.UTF8String ?: "?");
            return -1;
        }
        MLMultiArray *image = predict_chunk(ctx->model, chunk, states, &err);
        if (!image) {
            fail(error, error_size, "predict: %s", err.localizedDescription.UTF8String ?: "?");
            return -1;
        }
        const int out_frame_in_chunk = t_idx - chunk_start;
        const int pixel_h = h * 16;
        const int pixel_w = w * 16;
        *out_pw = pixel_w;
        *out_ph = pixel_h;
        if (image.dataType != MLMultiArrayDataTypeFloat16) {
            fail(error, error_size, "image not fp16");
            return -1;
        }
        log_array_strides(image, "image");
        const int img_t = (int)image.shape[1].intValue;
        const int img_h = (int)image.shape[2].intValue;
        const int img_w = (int)image.shape[3].intValue;
        if (out_frame_in_chunk >= img_t) {
            fail(error, error_size, "frame index oob");
            return -1;
        }
        // Core ML / ANE outputs may use padded strides; never index dataPointer densely.
        for (int y = 0; y < pixel_h && y < img_h; y++) {
            for (int x = 0; x < pixel_w && x < img_w; x++) {
                for (int k = 0; k < 3; k++) {
                    const int idx[5] = {0, out_frame_in_chunk, y, x, k};
                    float v = (float)f16_at(image, idx);
                    v = fminf(fmaxf(v, 0.f), 1.f) * 255.f;
                    out_rgb[(y * pixel_w + x) * 3 + k] = (uint8_t)v;
                }
            }
        }
        return 0;
    }
}
