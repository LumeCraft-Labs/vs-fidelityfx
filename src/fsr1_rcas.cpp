#include "../include/vs_fidelityfx.h"
#include "../include/pixel_access.h"
#include "../include/ffx_fsr1.h"
#include <stdlib.h>
#include <string.h>

static inline PixelVec load_fp(const float * const *p, int stride,
                               int x, int y, int w, int h) {
    x = clamp_coord(x, w);
    y = clamp_coord(y, h);
    int idx = y * stride + x;
    return pv_set(p[0][idx], p[1][idx], p[2][idx]);
}

static void fsr_rcas_con(FfxUInt32x4 con, FfxFloat32 sharpness) {
    sharpness = exp2f(-sharpness);
    con[0] = ffxAsUInt32(sharpness);
    con[1] = ffxPackHalf2x16((FfxFloat32x2){sharpness, sharpness});
    con[2] = 0;
    con[3] = 0;
}

static void fsr_rcas_f(PixelVec *result,
                       int x, int y,
                       const float * const *fp, int fp_stride, int fp_w, int fp_h,
                       const FfxUInt32x4 con) {
    // Load 5-tap cross pattern directly from float planes
    PixelVec pb = load_fp(fp, fp_stride, x, y - 1, fp_w, fp_h);
    PixelVec pd = load_fp(fp, fp_stride, x - 1, y, fp_w, fp_h);
    PixelVec pe = load_fp(fp, fp_stride, x, y, fp_w, fp_h);
    PixelVec pf = load_fp(fp, fp_stride, x + 1, y, fp_w, fp_h);
    PixelVec ph = load_fp(fp, fp_stride, x, y + 1, fp_w, fp_h);

    // Luma times 2 (scalar)
    float bL = pv_luma(pb);
    float dL = pv_luma(pd);
    float eL = pv_luma(pe);
    float fL = pv_luma(pf);
    float hL = pv_luma(ph);

    // Noise detection (scalar)
    float nz = 0.25f * bL + 0.25f * dL + 0.25f * fL + 0.25f * hL - eL;
    nz = ffxSaturate(ffxAbs(nz) * ffxApproximateReciprocalMedium(
        ffxMax3(ffxMax3(bL, dL, eL), fL, hL) - ffxMin3(ffxMin3(bL, dL, eL), fL, hL)));
    nz = -0.5f * nz + 1.0f;

    // Min and max of ring — parallel
    PixelVec mn4 = pv_min(pv_min(pv_min3(pb, pd, pf), ph), pe);
    PixelVec mx4 = pv_max(pv_max(pv_max3(pb, pd, pf), ph), pe);

    // Limiters
    PixelVec mn4_ring = pv_min(pv_min3(pb, pd, pf), ph);
    PixelVec mx4_ring = pv_max(pv_max3(pb, pd, pf), ph);

    const float lowerLimiterMultiplier = ffxSaturate(eL / ffxMin(ffxMin3(bL, dL, fL), hL));

    // hitMin = mn4 * rcp(4*mx4) * lowerLimiterMultiplier
    PixelVec four = pv_set1(4.0f);
    PixelVec mx4x4 = pv_mul(mx4_ring, four);
    PixelVec rcp_mx = pv_rcp_nr(mx4x4);
    PixelVec hitMin = pv_mul(pv_mul(mn4_ring, rcp_mx), pv_set1(lowerLimiterMultiplier));

    // hitMax = (1 - mx4) * rcp(4*mn4 - 4)
    PixelVec one = pv_set1(1.0f);
    PixelVec mn4x4m4 = pv_sub(pv_mul(mn4_ring, four), four);
    PixelVec rcp_mn = pv_rcp_nr(mn4x4m4);
    PixelVec hitMax = pv_mul(pv_sub(one, mx4_ring), rcp_mn);

    // lobe_rgb = max(-hitMin, hitMax)
    PixelVec neg_hitMin = pv_neg(hitMin);
    PixelVec lobe_rgb = pv_max(neg_hitMin, hitMax);

    // Extract max of lobe across R,G,B for scalar lobe
    float lobeMax = ffxMax3(pv_extract(lobe_rgb, 0), pv_extract(lobe_rgb, 1), pv_extract(lobe_rgb, 2));

    #define FSR_RCAS_LIMIT 0.1875f
    float lobe = ffxMax(-FSR_RCAS_LIMIT, ffxMin(lobeMax, 0.0f)) * ffxAsFloat(con[0]);

    // Resolve: (lobe*(b+d+h+f) + e) * rcpL
    float rcpL = ffxApproximateReciprocalMedium(4.0f * lobe + 1.0f);
    PixelVec vLobe = pv_set1(lobe);
    PixelVec vRcpL = pv_set1(rcpL);
    PixelVec ring_sum = pv_add(pv_add(pb, pd), pv_add(ph, pf));
    *result = pv_mul(pv_add(pv_mul(ring_sum, vLobe), pe), vRcpL);
}

static const VSFrame *VS_CC rcas_get_frame(int n, int activationReason, void *instanceData,
                                           void **frameData, VSFrameContext *frameCtx,
                                           VSCore *core, const VSAPI *vsapi) {
    RcasData *d = (RcasData *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        int width = vsapi->getFrameWidth(src, 0);
        int height = vsapi->getFrameHeight(src, 0);

        VSFrame *dst = vsapi->newVideoFrame(fi, width, height, src, core);

        // Pre-convert entire input to float planes
        PixelLoadContext ctx;
        init_pixel_context(&ctx, src, vsapi);

        float *fp[3];
        int fp_stride;
        float *fp_buf = convert_to_float_planes(fp, &fp_stride, &ctx);

        PixelStoreContext sctx;
        init_store_context(&sctx, dst, vsapi);

        const float * const *fp_c = (const float * const *)fp;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                PixelVec result;
                fsr_rcas_f(&result, x, y, fp_c, fp_stride, width, height, d->constants);
                store_pixel_vec(&sctx, x, y, result);
            }
        }

        free(fp_buf);
        vsapi->freeFrame(src);
        return dst;
    }

    return NULL;
}

static void VS_CC rcas_free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    RcasData *d = (RcasData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

void rcas_create(const VSMap *in, VSMap *out, void *userData,
                VSCore *core, const VSAPI *vsapi) {
    RcasData d;
    RcasData *data;
    int err;

    d.node = vsapi->mapGetNode(in, "clip", 0, 0);
    d.vi = vsapi->getVideoInfo(d.node);

    if (!vsh::isConstantVideoFormat(d.vi) ||
        (d.vi->format.colorFamily != cfRGB && d.vi->format.colorFamily != cfGray)) {
        vsapi->mapSetError(out, "RCAS: input clip must be RGB or GRAY format");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.vi->format.sampleType == stInteger) {
        if (d.vi->format.bitsPerSample < 8 || d.vi->format.bitsPerSample > 16) {
            vsapi->mapSetError(out, "RCAS: integer formats must be 8-16 bit");
            vsapi->freeNode(d.node);
            return;
        }
    } else if (d.vi->format.sampleType == stFloat) {
        if (d.vi->format.bitsPerSample != 16 && d.vi->format.bitsPerSample != 32) {
            vsapi->mapSetError(out, "RCAS: float formats must be 16-bit (half) or 32-bit");
            vsapi->freeNode(d.node);
            return;
        }
    } else {
        vsapi->mapSetError(out, "RCAS: unsupported sample type");
        vsapi->freeNode(d.node);
        return;
    }

    d.sharpness = (float)vsapi->mapGetFloat(in, "sharpness", 0, &err);
    if (err) d.sharpness = 0.0f;

    if (d.sharpness < 0.0f || d.sharpness > 2.0f) {
        vsapi->mapSetError(out, "RCAS: sharpness must be in range [0.0, 2.0]");
        vsapi->freeNode(d.node);
        return;
    }

    fsr_rcas_con(d.constants, d.sharpness);

    data = (RcasData *)malloc(sizeof(d));
    *data = d;

    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "RCAS", d.vi, rcas_get_frame, rcas_free,
                            fmParallel, deps, 1, data, core);
}
