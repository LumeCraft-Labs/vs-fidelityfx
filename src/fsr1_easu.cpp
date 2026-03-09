#include "../include/vs_fidelityfx.h"
#include "../include/pixel_access.h"
#include "../include/ffx_fsr1.h"
#include <stdlib.h>
#include <string.h>

static inline void fsr_easu_tap(
    PixelVec *aC, float *aW,
    float pixelOffset_x, float pixelOffset_y,
    float dir_x, float dir_y,
    float len_x, float len_y,
    float lob, float clp,
    PixelVec color)
{
    float vX = pixelOffset_x * dir_x + pixelOffset_y * dir_y;
    float vY = pixelOffset_x * (-dir_y) + pixelOffset_y * dir_x;
    vX *= len_x;
    vY *= len_y;
    float d2 = vX * vX + vY * vY;
    d2 = ffxMin(d2, clp);

    // Lanczos2 approximation
    float wB = 2.0f / 5.0f * d2 - 1.0f;
    float wA = lob * d2 - 1.0f;
    wB *= wB;
    wA *= wA;
    wB = 25.0f / 16.0f * wB - (25.0f / 16.0f - 1.0f);
    float w = wB * wA;

    PixelVec vw = pv_set1(w);
    *aC = pv_add(*aC, pv_mul(color, vw));
    *aW += w;
}

// Helper: fsrEasuSetFloat
static void fsr_easu_set_float(
    float *dir_x, float *dir_y, float *len,
    float pp_x, float pp_y,
    int biS, int biT, int biU, int biV,
    float lA, float lB, float lC, float lD, float lE)
{
    // Compute bilinear weight
    float w = 0.0f;
    if (biS) w = (1.0f - pp_x) * (1.0f - pp_y);
    if (biT) w = pp_x * (1.0f - pp_y);
    if (biU) w = (1.0f - pp_x) * pp_y;
    if (biV) w = pp_x * pp_y;

    // X direction
    float dc = lD - lC;
    float cb = lC - lB;
    float lenX = ffxMax(ffxAbs(dc), ffxAbs(cb));
    // Guard against divide-by-zero: on GPU rcp(0)=INF then saturate(0*INF=NaN)=0, but on CPU NaN propagates.
    lenX = lenX > 0.0f ? ffxReciprocal(lenX) : 0.0f;
    float dirX = lD - lB;
    *dir_x += dirX * w;
    lenX = ffxSaturateSafe(ffxAbs(dirX) * lenX);
    lenX *= lenX;
    *len += lenX * w;

    // Y direction
    float ec = lE - lC;
    float ca = lC - lA;
    float lenY = ffxMax(ffxAbs(ec), ffxAbs(ca));
    lenY = lenY > 0.0f ? ffxReciprocal(lenY) : 0.0f;
    float dirY = lE - lA;
    *dir_y += dirY * w;
    lenY = ffxSaturateSafe(ffxAbs(dirY) * lenY);
    lenY *= lenY;
    *len += lenY * w;
}

static void fsr_easu_float(
    PixelVec *pix,
    int out_x, int out_y,
    const PixelLoadContext *ctx,
    const FfxUInt32x4 con0, const FfxUInt32x4 con1,
    const FfxUInt32x4 con2, const FfxUInt32x4 con3)
{
    // Get position of 'f'
    float pp_x = out_x * ffxAsFloat(con0[0]) + ffxAsFloat(con0[2]);
    float pp_y = out_y * ffxAsFloat(con0[1]) + ffxAsFloat(con0[3]);
    float fp_x = floorf(pp_x);
    float fp_y = floorf(pp_y);
    pp_x -= fp_x;
    pp_y -= fp_y;

    int base_x = (int)fp_x;
    int base_y = (int)fp_y;

    // Load 12 pixels as PixelVec {R, G, B, 0}
    PixelVec pb = load_pixel_vec(ctx, base_x + 0, base_y - 1);
    PixelVec pc = load_pixel_vec(ctx, base_x + 1, base_y - 1);
    PixelVec pe = load_pixel_vec(ctx, base_x - 1, base_y + 0);
    PixelVec pf = load_pixel_vec(ctx, base_x + 0, base_y + 0);
    PixelVec pg = load_pixel_vec(ctx, base_x + 1, base_y + 0);
    PixelVec ph = load_pixel_vec(ctx, base_x + 2, base_y + 0);
    PixelVec pi = load_pixel_vec(ctx, base_x - 1, base_y + 1);
    PixelVec pj = load_pixel_vec(ctx, base_x + 0, base_y + 1);
    PixelVec pk = load_pixel_vec(ctx, base_x + 1, base_y + 1);
    PixelVec pl = load_pixel_vec(ctx, base_x + 2, base_y + 1);
    PixelVec pn = load_pixel_vec(ctx, base_x + 0, base_y + 2);
    PixelVec po = load_pixel_vec(ctx, base_x + 1, base_y + 2);

    // Compute luma for each pixel
    float bL = pv_luma(pb), cL = pv_luma(pc);
    float eL = pv_luma(pe), fL = pv_luma(pf);
    float gL = pv_luma(pg), hL = pv_luma(ph);
    float iL = pv_luma(pi), jL = pv_luma(pj);
    float kL = pv_luma(pk), lL = pv_luma(pl);
    float nL = pv_luma(pn), oL = pv_luma(po);

    // Accumulate direction and length (scalar — luma only)
    float dir_x = 0.0f, dir_y = 0.0f, len = 0.0f;
    fsr_easu_set_float(&dir_x, &dir_y, &len, pp_x, pp_y, 1, 0, 0, 0, bL, eL, fL, gL, jL);
    fsr_easu_set_float(&dir_x, &dir_y, &len, pp_x, pp_y, 0, 1, 0, 0, cL, fL, gL, hL, kL);
    fsr_easu_set_float(&dir_x, &dir_y, &len, pp_x, pp_y, 0, 0, 1, 0, fL, iL, jL, kL, nL);
    fsr_easu_set_float(&dir_x, &dir_y, &len, pp_x, pp_y, 0, 0, 0, 1, gL, jL, kL, lL, oL);

    // Normalize direction
    float dir2 = dir_x * dir_x + dir_y * dir_y;
    int zro = dir2 < (1.0f / 32768.0f);
    float dirR = ffxRsqrt(dir2);
    dirR = zro ? 1.0f : dirR;
    dir_x = zro ? 1.0f : dir_x;
    dir_y = zro ? 0.0f : dir_y;
    dir_x *= dirR;
    dir_y *= dirR;

    // Transform length
    len = len * 0.5f;
    len *= len;

    // Anisotropic stretch
    float maxDir = ffxMax(ffxAbs(dir_x), ffxAbs(dir_y));
    float stretch = maxDir > 0.0f
        ? (dir_x * dir_x + dir_y * dir_y) * ffxReciprocal(maxDir)
        : 1.0f;
    float len2_x = 1.0f + (stretch - 1.0f) * len;
    float len2_y = 1.0f - 0.5f * len;

    // Lobe and clipping
    float lob = 0.5f + (1.0f / 4.0f - 0.04f - 0.5f) * len;
    float clp = ffxReciprocal(lob);

    // Min/max of 4 nearest (f, g, j, k) — parallel across R,G,B
    PixelVec min4 = pv_min(pv_min(pv_min(pf, pg), pj), pk);
    PixelVec max4 = pv_max(pv_max(pv_max(pf, pg), pj), pk);

    // Accumulate 12 taps — parallel across R,G,B
    PixelVec aC = pv_zero();
    float aW = 0.0f;
    fsr_easu_tap(&aC, &aW, 0.0f - pp_x, -1.0f - pp_y, dir_x, dir_y, len2_x, len2_y, lob, clp, pb);
    fsr_easu_tap(&aC, &aW, 1.0f - pp_x, -1.0f - pp_y, dir_x, dir_y, len2_x, len2_y, lob, clp, pc);
    fsr_easu_tap(&aC, &aW, -1.0f - pp_x, 1.0f - pp_y, dir_x, dir_y, len2_x, len2_y, lob, clp, pi);
    fsr_easu_tap(&aC, &aW, 0.0f - pp_x, 1.0f - pp_y, dir_x, dir_y, len2_x, len2_y, lob, clp, pj);
    fsr_easu_tap(&aC, &aW, 0.0f - pp_x, 0.0f - pp_y, dir_x, dir_y, len2_x, len2_y, lob, clp, pf);
    fsr_easu_tap(&aC, &aW, -1.0f - pp_x, 0.0f - pp_y, dir_x, dir_y, len2_x, len2_y, lob, clp, pe);
    fsr_easu_tap(&aC, &aW, 1.0f - pp_x, 1.0f - pp_y, dir_x, dir_y, len2_x, len2_y, lob, clp, pk);
    fsr_easu_tap(&aC, &aW, 2.0f - pp_x, 1.0f - pp_y, dir_x, dir_y, len2_x, len2_y, lob, clp, pl);
    fsr_easu_tap(&aC, &aW, 2.0f - pp_x, 0.0f - pp_y, dir_x, dir_y, len2_x, len2_y, lob, clp, ph);
    fsr_easu_tap(&aC, &aW, 1.0f - pp_x, 0.0f - pp_y, dir_x, dir_y, len2_x, len2_y, lob, clp, pg);
    fsr_easu_tap(&aC, &aW, 1.0f - pp_x, 2.0f - pp_y, dir_x, dir_y, len2_x, len2_y, lob, clp, po);
    fsr_easu_tap(&aC, &aW, 0.0f - pp_x, 2.0f - pp_y, dir_x, dir_y, len2_x, len2_y, lob, clp, pn);

    // Normalize and dering
    if (!(aW >= 1e-6f)) {
        *pix = pf;
    } else {
        float rcpW = ffxReciprocal(aW);
        rcpW = ffxMin(rcpW, 1000.0f);
        PixelVec vRcpW = pv_set1(rcpW);
        PixelVec result = pv_mul(aC, vRcpW);
        // Clamp to [min4, max4]
        result = pv_max(min4, result);
        result = pv_min(max4, result);
        *pix = result;
    }
}

static const VSFrame *VS_CC easu_get_frame(int n, int activationReason, void *instanceData,
                                           void **frameData, VSFrameContext *frameCtx,
                                           VSCore *core, const VSAPI *vsapi) {
    EasuData *d = (EasuData *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);

        VSFrame *dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);

        PixelLoadContext ctx;
        init_pixel_context(&ctx, src, vsapi);

        PixelStoreContext sctx;
        init_store_context(&sctx, dst, vsapi);

        for (int y = 0; y < d->vi.height; y++) {
            for (int x = 0; x < d->vi.width; x++) {
                PixelVec result;
                fsr_easu_float(&result, x, y, &ctx,
                              d->con0, d->con1, d->con2, d->con3);
                store_pixel_vec(&sctx, x, y, result);
            }
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return NULL;
}

static void VS_CC easu_free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    EasuData *d = (EasuData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

void easu_create(const VSMap *in, VSMap *out, void *userData,
                VSCore *core, const VSAPI *vsapi) {
    EasuData d;
    EasuData *data;
    int err;

    d.node = vsapi->mapGetNode(in, "clip", 0, 0);
    const VSVideoInfo *vi = vsapi->getVideoInfo(d.node);

    if (!vsh::isConstantVideoFormat(vi) ||
        (vi->format.colorFamily != cfRGB && vi->format.colorFamily != cfGray)) {
        vsapi->mapSetError(out, "EASU: input clip must be RGB or GRAY format");
        vsapi->freeNode(d.node);
        return;
    }

    if (vi->format.sampleType == stInteger) {
        if (vi->format.bitsPerSample < 8 || vi->format.bitsPerSample > 16) {
            vsapi->mapSetError(out, "EASU: integer formats must be 8-16 bit");
            vsapi->freeNode(d.node);
            return;
        }
    } else if (vi->format.sampleType == stFloat) {
        if (vi->format.bitsPerSample != 16 && vi->format.bitsPerSample != 32) {
            vsapi->mapSetError(out, "EASU: float formats must be 16-bit (half) or 32-bit");
            vsapi->freeNode(d.node);
            return;
        }
    } else {
        vsapi->mapSetError(out, "EASU: unsupported sample type");
        vsapi->freeNode(d.node);
        return;
    }

    int width = (int)vsapi->mapGetInt(in, "width", 0, &err);
    int height = (int)vsapi->mapGetInt(in, "height", 0, &err);

    if (width < vi->width || height < vi->height) {
        vsapi->mapSetError(out, "EASU: output dimensions must be >= input dimensions");
        vsapi->freeNode(d.node);
        return;
    }

    d.input_width = vi->width;
    d.input_height = vi->height;

    d.vi = *vi;
    d.vi.width = width;
    d.vi.height = height;

    ffxFsrPopulateEasuConstants(
        d.con0, d.con1, d.con2, d.con3,
        (float)vi->width, (float)vi->height,
        (float)vi->width, (float)vi->height,
        (float)width, (float)height);

    data = (EasuData *)malloc(sizeof(d));
    *data = d;

    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "EASU", &data->vi, easu_get_frame, easu_free,
                            fmParallel, deps, 1, data, core);
}
