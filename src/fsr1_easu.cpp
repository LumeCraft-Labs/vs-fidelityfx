#include "../include/vs_fidelityfx.h"
#include "../include/pixel_access.h"
#include "../include/ffx_fsr1.h"
#include <stdlib.h>
#include <string.h>

// ========== RGBL interleaved pixel access (1 load = RGB + Luma) ==============

// Load pixel as PixelVec from interleaved RGBL buffer (16-byte aligned)
static inline PixelVec load_rgbl(const float *buf, int stride,
                                 int x, int y, int w, int h) {
    x = clamp_coord(x, w);
    y = clamp_coord(y, h);
#ifdef __SSE2__
    return _mm_load_ps(buf + ((size_t)y * stride + x) * 4);
#else
    const float *p = buf + ((size_t)y * stride + x) * 4;
    return pv_set(p[0], p[1], p[2]);
#endif
}

// Extract pre-computed luma from RGBL pixel (4th component)
static inline float luma_rgbl(const float *buf, int stride,
                              int x, int y, int w, int h) {
    x = clamp_coord(x, w);
    y = clamp_coord(y, h);
    return buf[((size_t)y * stride + x) * 4 + 3];
}

// =============================================================================

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
    const float *rgbl, int rgbl_stride, int fp_w, int fp_h,
    const FfxUInt32x4 con0)
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

    // Load 12 pixels from interleaved RGBL buffer (1 SSE load each)
    PixelVec pb = load_rgbl(rgbl, rgbl_stride, base_x + 0, base_y - 1, fp_w, fp_h);
    PixelVec pc = load_rgbl(rgbl, rgbl_stride, base_x + 1, base_y - 1, fp_w, fp_h);
    PixelVec pe = load_rgbl(rgbl, rgbl_stride, base_x - 1, base_y + 0, fp_w, fp_h);
    PixelVec pf = load_rgbl(rgbl, rgbl_stride, base_x + 0, base_y + 0, fp_w, fp_h);
    PixelVec pg = load_rgbl(rgbl, rgbl_stride, base_x + 1, base_y + 0, fp_w, fp_h);
    PixelVec ph = load_rgbl(rgbl, rgbl_stride, base_x + 2, base_y + 0, fp_w, fp_h);
    PixelVec pi = load_rgbl(rgbl, rgbl_stride, base_x - 1, base_y + 1, fp_w, fp_h);
    PixelVec pj = load_rgbl(rgbl, rgbl_stride, base_x + 0, base_y + 1, fp_w, fp_h);
    PixelVec pk = load_rgbl(rgbl, rgbl_stride, base_x + 1, base_y + 1, fp_w, fp_h);
    PixelVec pl = load_rgbl(rgbl, rgbl_stride, base_x + 2, base_y + 1, fp_w, fp_h);
    PixelVec pn = load_rgbl(rgbl, rgbl_stride, base_x + 0, base_y + 2, fp_w, fp_h);
    PixelVec po = load_rgbl(rgbl, rgbl_stride, base_x + 1, base_y + 2, fp_w, fp_h);

    // Extract pre-computed luma from 4th component (zero computation)
    float bL = pv_extract(pb, 3), cL = pv_extract(pc, 3);
    float eL = pv_extract(pe, 3), fL = pv_extract(pf, 3);
    float gL = pv_extract(pg, 3), hL = pv_extract(ph, 3);
    float iL = pv_extract(pi, 3), jL = pv_extract(pj, 3);
    float kL = pv_extract(pk, 3), lL = pv_extract(pl, 3);
    float nL = pv_extract(pn, 3), oL = pv_extract(po, 3);

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

// ================== Fast-mode EASU (reduced quality) =========================

static void fsr_easu_float_fast(
    PixelVec *pix,
    int out_x, int out_y,
    const float *rgbl, int rgbl_stride, int fp_w, int fp_h,
    const FfxUInt32x4 con0)
{
    float pp_x = out_x * ffxAsFloat(con0[0]) + ffxAsFloat(con0[2]);
    float pp_y = out_y * ffxAsFloat(con0[1]) + ffxAsFloat(con0[3]);
    float fp_x = floorf(pp_x);
    float fp_y = floorf(pp_y);
    pp_x -= fp_x;
    pp_y -= fp_y;

    int base_x = (int)fp_x;
    int base_y = (int)fp_y;

    // 5-tap cross pattern: A(top), B(left), C(center), D(right), E(bottom)
    PixelVec pA = load_rgbl(rgbl, rgbl_stride, base_x, base_y - 1, fp_w, fp_h);
    PixelVec pB = load_rgbl(rgbl, rgbl_stride, base_x - 1, base_y, fp_w, fp_h);
    PixelVec pC = load_rgbl(rgbl, rgbl_stride, base_x, base_y, fp_w, fp_h);
    PixelVec pD = load_rgbl(rgbl, rgbl_stride, base_x + 1, base_y, fp_w, fp_h);
    PixelVec pE = load_rgbl(rgbl, rgbl_stride, base_x, base_y + 1, fp_w, fp_h);

    float lA = pv_extract(pA, 3), lB = pv_extract(pB, 3), lC = pv_extract(pC, 3);
    float lD = pv_extract(pD, 3), lE = pv_extract(pE, 3);

    // Direction from cross differences
    float dc = lD - lC, cb = lC - lB;
    float lenX = ffxMax(ffxAbs(dc), ffxAbs(cb));
    lenX = lenX > 0.0f ? ffxReciprocal(lenX) : 0.0f;
    float dirX = lD - lB;
    lenX = ffxSaturateSafe(ffxAbs(dirX) * lenX);
    lenX *= lenX;

    float ec = lE - lC, ca = lC - lA;
    float lenY = ffxMax(ffxAbs(ec), ffxAbs(ca));
    lenY = lenY > 0.0f ? ffxReciprocal(lenY) : 0.0f;
    float dirY = lE - lA;
    lenY = ffxSaturateSafe(ffxAbs(dirY) * lenY);
    float len = lenY * lenY + lenX;

    float dir2 = dirX * dirX + dirY * dirY;

    // Early exit: flat region → bilinear
    if (dir2 < (1.0f / 64.0f)) {
        // Load the remaining 3 pixels for bilinear (f,g,j,k pattern)
        PixelVec pg = load_rgbl(rgbl, rgbl_stride, base_x + 1, base_y, fp_w, fp_h);
        PixelVec pj = load_rgbl(rgbl, rgbl_stride, base_x, base_y + 1, fp_w, fp_h);
        PixelVec pk = load_rgbl(rgbl, rgbl_stride, base_x + 1, base_y + 1, fp_w, fp_h);
        PixelVec row0 = pv_add(pv_mul(pC, pv_set1(1.0f - pp_x)),
                               pv_mul(pg, pv_set1(pp_x)));
        PixelVec row1 = pv_add(pv_mul(pj, pv_set1(1.0f - pp_x)),
                               pv_mul(pk, pv_set1(pp_x)));
        *pix = pv_add(pv_mul(row0, pv_set1(1.0f - pp_y)),
                      pv_mul(row1, pv_set1(pp_y)));
        return;
    }

    float dirR = ffxRsqrt(dir2);
    dirX *= dirR;
    dirY *= dirR;
    len *= 0.5f;
    len *= len;

    float stretch = (dirX * dirX + dirY * dirY)
                    * ffxReciprocal(ffxMax(ffxAbs(dirX), ffxAbs(dirY)));
    float len2_x = 1.0f + (stretch - 1.0f) * len;
    float len2_y = 1.0f - 0.5f * len;

    float lob = 0.5f + (1.0f / 4.0f - 0.04f - 0.5f) * len;
    float clp = ffxReciprocal(lob);

    // Now load remaining pixels for 12-tap
    PixelVec pb = pA;  // (base_x, base_y-1) already loaded as pA
    PixelVec pc = load_rgbl(rgbl, rgbl_stride, base_x + 1, base_y - 1, fp_w, fp_h);
    PixelVec pe = pB;  // (base_x-1, base_y) already loaded as pB
    PixelVec pf = pC;  // center
    PixelVec pg = pD;  // (base_x+1, base_y) already loaded as pD
    PixelVec ph = load_rgbl(rgbl, rgbl_stride, base_x + 2, base_y, fp_w, fp_h);
    PixelVec pi = load_rgbl(rgbl, rgbl_stride, base_x - 1, base_y + 1, fp_w, fp_h);
    PixelVec pj = pE;  // (base_x, base_y+1) already loaded as pE
    PixelVec pk = load_rgbl(rgbl, rgbl_stride, base_x + 1, base_y + 1, fp_w, fp_h);
    PixelVec pl = load_rgbl(rgbl, rgbl_stride, base_x + 2, base_y + 1, fp_w, fp_h);
    PixelVec pn = load_rgbl(rgbl, rgbl_stride, base_x, base_y + 2, fp_w, fp_h);
    PixelVec po = load_rgbl(rgbl, rgbl_stride, base_x + 1, base_y + 2, fp_w, fp_h);

    PixelVec min4 = pv_min(pv_min(pv_min(pf, pg), pj), pk);
    PixelVec max4 = pv_max(pv_max(pv_max(pf, pg), pj), pk);

    PixelVec aC = pv_zero();
    float aW = 0.0f;
    fsr_easu_tap(&aC, &aW, 0.0f - pp_x, -1.0f - pp_y, dirX, dirY, len2_x, len2_y, lob, clp, pb);
    fsr_easu_tap(&aC, &aW, 1.0f - pp_x, -1.0f - pp_y, dirX, dirY, len2_x, len2_y, lob, clp, pc);
    fsr_easu_tap(&aC, &aW, -1.0f - pp_x, 1.0f - pp_y, dirX, dirY, len2_x, len2_y, lob, clp, pi);
    fsr_easu_tap(&aC, &aW, 0.0f - pp_x, 1.0f - pp_y, dirX, dirY, len2_x, len2_y, lob, clp, pj);
    fsr_easu_tap(&aC, &aW, 0.0f - pp_x, 0.0f - pp_y, dirX, dirY, len2_x, len2_y, lob, clp, pf);
    fsr_easu_tap(&aC, &aW, -1.0f - pp_x, 0.0f - pp_y, dirX, dirY, len2_x, len2_y, lob, clp, pe);
    fsr_easu_tap(&aC, &aW, 1.0f - pp_x, 1.0f - pp_y, dirX, dirY, len2_x, len2_y, lob, clp, pk);
    fsr_easu_tap(&aC, &aW, 2.0f - pp_x, 1.0f - pp_y, dirX, dirY, len2_x, len2_y, lob, clp, pl);
    fsr_easu_tap(&aC, &aW, 2.0f - pp_x, 0.0f - pp_y, dirX, dirY, len2_x, len2_y, lob, clp, ph);
    fsr_easu_tap(&aC, &aW, 1.0f - pp_x, 0.0f - pp_y, dirX, dirY, len2_x, len2_y, lob, clp, pg);
    fsr_easu_tap(&aC, &aW, 1.0f - pp_x, 2.0f - pp_y, dirX, dirY, len2_x, len2_y, lob, clp, po);
    fsr_easu_tap(&aC, &aW, 0.0f - pp_x, 2.0f - pp_y, dirX, dirY, len2_x, len2_y, lob, clp, pn);

    if (!(aW >= 1e-6f)) {
        *pix = pf;
    } else {
        float rcpW = ffxReciprocal(aW);
        rcpW = ffxMin(rcpW, 1000.0f);
        PixelVec vRcpW = pv_set1(rcpW);
        PixelVec result = pv_mul(aC, vRcpW);
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

        // Pre-convert entire input to interleaved RGBL (R,G,B,Luma per pixel)
        PixelLoadContext ctx;
        init_pixel_context(&ctx, src, vsapi);

        int rgbl_stride;
        float *rgbl_buf = convert_to_rgbl(&rgbl_stride, &ctx);

        PixelStoreContext sctx;
        init_store_context(&sctx, dst, vsapi);

        for (int y = 0; y < d->vi.height; y++) {
            for (int x = 0; x < d->vi.width; x++) {
                PixelVec result;
                if (d->fast)
                    fsr_easu_float_fast(&result, x, y, rgbl_buf, rgbl_stride,
                                       ctx.width, ctx.height, d->con0);
                else
                    fsr_easu_float(&result, x, y, rgbl_buf, rgbl_stride,
                                  ctx.width, ctx.height, d->con0);
                sctx.store_vec(&sctx, x, y, result);
            }
        }

#ifdef _WIN32
        _aligned_free(rgbl_buf);
#else
        free(rgbl_buf);
#endif
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

    int fast = (int)vsapi->mapGetInt(in, "fast", 0, &err);
    if (err) fast = 0;
    d.fast = fast;

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
