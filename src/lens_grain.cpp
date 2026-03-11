#include "../include/vs_fidelityfx.h"
#include "../include/pixel_access.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// PCG3D 16-bit hash — noise basis for film grain
static void pcg3d16(uint32_t v[3]) {
    v[0] = v[0] * 12829u + 47989u;
    v[1] = v[1] * 12829u + 47989u;
    v[2] = v[2] * 12829u + 47989u;
    v[0] += v[1] * v[2];
    v[1] += v[2] * v[0];
    v[2] += v[0] * v[1];
    v[0] += v[1] * v[2];
    v[1] += v[2] * v[0];
    v[2] += v[0] * v[1];
    v[0] >>= 16u;
    v[1] >>= 16u;
    v[2] >>= 16u;
}

// Convert PCG output to [-0.5, 0.5] float
static inline float pcg_to_float(uint32_t val) {
    return (float)val * (1.0f / 65536.0f) - 0.5f;
}

// 2D Simplex noise grid transform (must stay at float32 precision to avoid artifacts)
static void simplex(float result[2], float px, float py) {
    const float F2 = 0.36602540378f;   // (sqrt(3) - 1) / 2
    const float G2 = 0.21132486540f;   // (3 - sqrt(3)) / 6

    float u = (px + py) * F2;
    float piX = roundf(px + u);
    float piY = roundf(py + u);
    float v = (piX + piY) * G2;
    float p0X = piX - v;
    float p0Y = piY - v;
    result[0] = px - p0X;
    result[1] = py - p0Y;
}

static const VSFrame *VS_CC grain_get_frame(int n, int activationReason, void *instanceData,
                                            void **frameData, VSFrameContext *frameCtx,
                                            VSCore *core, const VSAPI *vsapi) {
    GrainData *d = (GrainData *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        int width = vsapi->getFrameWidth(src, 0);
        int height = vsapi->getFrameHeight(src, 0);

        VSFrame *dst = vsapi->newVideoFrame(fi, width, height, src, core);

        // amount=0: no grain, just copy
        if (d->amount <= 0.0f) {
            for (int p = 0; p < fi->numPlanes; p++) {
                const uint8_t *srcp = vsapi->getReadPtr(src, p);
                uint8_t *dstp = vsapi->getWritePtr(dst, p);
                ptrdiff_t stride = vsapi->getStride(src, p);
                int rowsize = vsapi->getFrameWidth(src, p) * fi->bytesPerSample;
                for (int row = 0; row < vsapi->getFrameHeight(src, p); row++)
                    memcpy(dstp + row * stride, srcp + row * stride, rowsize);
            }
            vsapi->freeFrame(src);
            return dst;
        }

        PixelLoadContext ctx;
        init_pixel_context(&ctx, src, vsapi);

        float *fp[3];
        int fp_stride;
        float *fp_buf = convert_to_float_planes(fp, &fp_stride, &ctx);

        PixelStoreContext sctx;
        init_store_context(&sctx, dst, vsapi);

        // Seed: use frame number by default, or user-specified fixed seed
        uint32_t grainSeed = (d->seed >= 0) ? (uint32_t)d->seed : (uint32_t)n;
        float grainScale = d->scale;
        float grainAmount = d->amount;
        float invScale = 1.0f / grainScale;
        float invScaleCoarse = 8.0f / grainScale;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // PCG hash for randomization
                uint32_t pcg[3];
                pcg[0] = (uint32_t)((float)x * invScaleCoarse);
                pcg[1] = (uint32_t)((float)y * invScaleCoarse);
                pcg[2] = grainSeed;
                pcg3d16(pcg);

                float randX = pcg_to_float(pcg[0]);
                float randY = pcg_to_float(pcg[1]);

                // Simplex noise
                float simplexP[2];
                simplex(simplexP, (float)x * invScale + randX, (float)y * invScale + randY);

                // Grain shape: exponential decay from simplex magnitude
                float len = sqrtf(simplexP[0] * simplexP[0] + simplexP[1] * simplexP[1]);
                float grain = 1.0f - 2.0f * exp2f(-len * 3.0f);

                // Apply grain: read from pre-converted float planes
                int idx = y * fp_stride + x;
                float rgb[3];
                rgb[0] = fp[0][idx];
                rgb[1] = fp[1][idx];
                rgb[2] = fp[2][idx];
                for (int c = 0; c < 3; c++) {
                    float limit = fminf(rgb[c], 1.0f - rgb[c]);
                    rgb[c] += grain * limit * grainAmount;
                }
                sctx.store_rgb(&sctx, x, y, rgb);
            }
        }

        free(fp_buf);
        vsapi->freeFrame(src);
        return dst;
    }

    return NULL;
}

static void VS_CC grain_free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    GrainData *d = (GrainData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

void grain_create(const VSMap *in, VSMap *out, void *userData,
                 VSCore *core, const VSAPI *vsapi) {
    GrainData d;
    int err;

    d.node = vsapi->mapGetNode(in, "clip", 0, 0);
    d.vi = vsapi->getVideoInfo(d.node);

    if (!vsh::isConstantVideoFormat(d.vi) ||
        (d.vi->format.colorFamily != cfRGB && d.vi->format.colorFamily != cfGray)) {
        vsapi->mapSetError(out, "Grain: input must be RGB or GRAY format");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.vi->format.sampleType == stInteger) {
        if (d.vi->format.bitsPerSample < 8 || d.vi->format.bitsPerSample > 16) {
            vsapi->mapSetError(out, "Grain: integer formats must be 8-16 bit");
            vsapi->freeNode(d.node);
            return;
        }
    } else if (d.vi->format.sampleType == stFloat) {
        if (d.vi->format.bitsPerSample != 16 && d.vi->format.bitsPerSample != 32) {
            vsapi->mapSetError(out, "Grain: float formats must be 16-bit (half) or 32-bit");
            vsapi->freeNode(d.node);
            return;
        }
    }

    d.scale = (float)vsapi->mapGetFloat(in, "scale", 0, &err);
    if (err) d.scale = 1.0f;

    d.amount = (float)vsapi->mapGetFloat(in, "amount", 0, &err);
    if (err) d.amount = 0.05f;

    d.seed = (int)vsapi->mapGetInt(in, "seed", 0, &err);
    if (err) d.seed = -1;  // -1 = use frame number

    if (d.scale < 0.01f || d.scale > 20.0f) {
        vsapi->mapSetError(out, "Grain: scale must be in range [0.01, 20.0]");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.amount < 0.0f || d.amount > 20.0f) {
        vsapi->mapSetError(out, "Grain: amount must be in range [0.0, 20.0]");
        vsapi->freeNode(d.node);
        return;
    }

    GrainData *data = (GrainData *)malloc(sizeof(d));
    *data = d;

    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "Grain", d.vi, grain_get_frame, grain_free,
                            fmParallel, deps, 1, data, core);
}
