#include "../include/vs_fidelityfx.h"
#include "../include/pixel_access.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const VSFrame *VS_CC vignette_get_frame(int n, int activationReason, void *instanceData,
                                               void **frameData, VSFrameContext *frameCtx,
                                               VSCore *core, const VSAPI *vsapi) {
    VignetteData *d = (VignetteData *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        int width = vsapi->getFrameWidth(src, 0);
        int height = vsapi->getFrameHeight(src, 0);

        VSFrame *dst = vsapi->newVideoFrame(fi, width, height, src, core);

        // intensity=0: no darkening, just copy
        if (d->intensity <= 0.0f) {
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

        float centerX = (float)(width / 2);
        float centerY = (float)(height / 2);
        float piOver4 = (float)(M_PI * 0.25);

        for (int y = 0; y < height; y++) {
            // Precompute Y component of vignette mask for this row
            float fromCenterY = fabsf((float)y - centerY) / centerY;
            float maskY = cosf(fromCenterY * d->intensity * piOver4);
            maskY = maskY * maskY;
            maskY = maskY * maskY;

            for (int x = 0; x < width; x++) {
                float fromCenterX = fabsf((float)x - centerX) / centerX;
                float maskX = cosf(fromCenterX * d->intensity * piOver4);
                maskX = maskX * maskX;
                maskX = maskX * maskX;

                float mask = maskX * maskY;
                if (mask < 0.0f) mask = 0.0f;
                if (mask > 1.0f) mask = 1.0f;

                float rgb[3];
                load_pixel_rgb(rgb, &ctx, x, y);
                rgb[0] *= mask;
                rgb[1] *= mask;
                rgb[2] *= mask;
                store_pixel_rgb(dst, vsapi, x, y, rgb);
            }
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return NULL;
}

static void VS_CC vignette_free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    VignetteData *d = (VignetteData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

void vignette_create(const VSMap *in, VSMap *out, void *userData,
                    VSCore *core, const VSAPI *vsapi) {
    VignetteData d;
    int err;

    d.node = vsapi->mapGetNode(in, "clip", 0, 0);
    d.vi = vsapi->getVideoInfo(d.node);

    if (!vsh::isConstantVideoFormat(d.vi) ||
        (d.vi->format.colorFamily != cfRGB && d.vi->format.colorFamily != cfGray)) {
        vsapi->mapSetError(out, "Vignette: input must be RGB or GRAY format");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.vi->format.sampleType == stInteger) {
        if (d.vi->format.bitsPerSample < 8 || d.vi->format.bitsPerSample > 16) {
            vsapi->mapSetError(out, "Vignette: integer formats must be 8-16 bit");
            vsapi->freeNode(d.node);
            return;
        }
    } else if (d.vi->format.sampleType == stFloat) {
        if (d.vi->format.bitsPerSample != 16 && d.vi->format.bitsPerSample != 32) {
            vsapi->mapSetError(out, "Vignette: float formats must be 16-bit (half) or 32-bit");
            vsapi->freeNode(d.node);
            return;
        }
    }

    d.intensity = (float)vsapi->mapGetFloat(in, "intensity", 0, &err);
    if (err) d.intensity = 1.0f;

    if (d.intensity < 0.0f || d.intensity > 2.0f) {
        vsapi->mapSetError(out, "Vignette: intensity must be in range [0.0, 2.0]");
        vsapi->freeNode(d.node);
        return;
    }

    VignetteData *data = (VignetteData *)malloc(sizeof(d));
    *data = d;

    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "Vignette", d.vi, vignette_get_frame, vignette_free,
                            fmParallel, deps, 1, data, core);
}
