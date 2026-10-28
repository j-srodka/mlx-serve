#import "h3_tae_preview.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

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
    return image;
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
        const f16 *img = image.dataPointer;
        const int img_t = (int)image.shape[1].intValue;
        const int img_h = (int)image.shape[2].intValue;
        const int img_w = (int)image.shape[3].intValue;
        if (out_frame_in_chunk >= img_t) {
            fail(error, error_size, "frame index oob");
            return -1;
        }
        for (int y = 0; y < pixel_h && y < img_h; y++) {
            for (int x = 0; x < pixel_w && x < img_w; x++) {
                const size_t base = (size_t)(((out_frame_in_chunk * img_h + y) * img_w + x) * 3);
                for (int k = 0; k < 3; k++) {
                    float v = (float)img[base + (size_t)k];
                    v = fminf(fmaxf(v, 0.f), 1.f) * 255.f;
                    out_rgb[(y * pixel_w + x) * 3 + k] = (uint8_t)v;
                }
            }
        }
        return 0;
    }
}
