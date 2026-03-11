#include "../include/vs_fidelityfx.h"
#include "../include/pixel_access.h"
#include <stdlib.h>
#include <math.h>

static void lens_get_rg_mag(float intensity, float *redMag, float *greenMag) {
    const float A = 1.5220f;               // K5 glass constant
    const float B = 0.00459f * intensity;   // µm²

    const float redWL   = 0.612f;  // red wavelength µm
    const float greenWL = 0.549f;  // green wavelength µm
    const float blueWL  = 0.464f;  // blue wavelength µm

    const float redIdx   = A + B / (redWL * redWL);
    const float greenIdx = A + B / (greenWL * greenWL);
    const float blueIdx  = A + B / (blueWL * blueWL);

    *redMag   = (redIdx - 1.0f) / (blueIdx - 1.0f);
    *greenMag = (greenIdx - 1.0f) / (blueIdx - 1.0f);
}

static const VSFrame *VS_CC ca_get_frame(int n, int activationReason, void *instanceData,
                                         void **frameData, VSFrameContext *frameCtx,
                                         VSCore *core, const VSAPI *vsapi) {
    ChromaticAberrationData *d = (ChromaticAberrationData *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        int width = vsapi->getFrameWidth(src, 0);
        int height = vsapi->getFrameHeight(src, 0);

        VSFrame *dst = vsapi->newVideoFrame(fi, width, height, src, core);

        // intensity=0: no shift, just copy
        if (d->intensity <= 0.0f) {
            for (int p = 0; p < fi->numPlanes; p++) {
                const uint8_t *srcp = vsapi->getReadPtr(src, p);
                uint8_t *dstp = vsapi->getWritePtr(dst, p);
                ptrdiff_t stride = vsapi->getStride(src, p);
                int rowsize = vsapi->getFrameWidth(src, p) * fi->bytesPerSample;
                for (int y = 0; y < vsapi->getFrameHeight(src, p); y++)
                    memcpy(dstp + y * stride, srcp + y * stride, rowsize);
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

        float redMag, greenMag;
        lens_get_rg_mag(d->intensity, &redMag, &greenMag);

        float centerX = (float)(width / 2);
        float centerY = (float)(height / 2);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Red channel: sample at offset position
                float redShiftX = ((float)x - centerX) * redMag + centerX;
                float redShiftY = ((float)y - centerY) * redMag + centerY;

                // Green channel: sample at offset position
                float greenShiftX = ((float)x - centerX) * greenMag + centerX;
                float greenShiftY = ((float)y - centerY) * greenMag + centerY;

                // Bilinear sample from pre-converted float planes
                // Red channel
                float rx = fminf(fmaxf(redShiftX, 0.0f), (float)(width - 1));
                float ry = fminf(fmaxf(redShiftY, 0.0f), (float)(height - 1));
                int rx0 = (int)rx, ry0 = (int)ry;
                int rx1 = rx0 < width - 1 ? rx0 + 1 : rx0;
                int ry1 = ry0 < height - 1 ? ry0 + 1 : ry0;
                float rfx = rx - rx0, rfy = ry - ry0;
                float red = fp[0][ry0 * fp_stride + rx0] * (1.0f - rfx) * (1.0f - rfy)
                          + fp[0][ry0 * fp_stride + rx1] * rfx * (1.0f - rfy)
                          + fp[0][ry1 * fp_stride + rx0] * (1.0f - rfx) * rfy
                          + fp[0][ry1 * fp_stride + rx1] * rfx * rfy;

                // Green channel
                float gx = fminf(fmaxf(greenShiftX, 0.0f), (float)(width - 1));
                float gy = fminf(fmaxf(greenShiftY, 0.0f), (float)(height - 1));
                int gx0 = (int)gx, gy0 = (int)gy;
                int gx1 = gx0 < width - 1 ? gx0 + 1 : gx0;
                int gy1 = gy0 < height - 1 ? gy0 + 1 : gy0;
                float gfx = gx - gx0, gfy = gy - gy0;
                float green = fp[1][gy0 * fp_stride + gx0] * (1.0f - gfx) * (1.0f - gfy)
                            + fp[1][gy0 * fp_stride + gx1] * gfx * (1.0f - gfy)
                            + fp[1][gy1 * fp_stride + gx0] * (1.0f - gfx) * gfy
                            + fp[1][gy1 * fp_stride + gx1] * gfx * gfy;

                // Blue channel: at original position (no shift)
                float blue = fp[2][y * fp_stride + x];

                float rgb[3] = { red, green, blue };
                sctx.store_rgb(&sctx, x, y, rgb);
            }
        }

        free(fp_buf);
        vsapi->freeFrame(src);
        return dst;
    }

    return NULL;
}

static void VS_CC ca_free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    ChromaticAberrationData *d = (ChromaticAberrationData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

void chromatic_aberration_create(const VSMap *in, VSMap *out, void *userData,
                                VSCore *core, const VSAPI *vsapi) {
    ChromaticAberrationData d;
    int err;

    d.node = vsapi->mapGetNode(in, "clip", 0, 0);
    d.vi = vsapi->getVideoInfo(d.node);

    // RGB only — chromatic aberration has no meaning for grayscale
    if (!vsh::isConstantVideoFormat(d.vi) || d.vi->format.colorFamily != cfRGB) {
        vsapi->mapSetError(out, "ChromaticAberration: input must be RGB format");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.vi->format.sampleType == stInteger) {
        if (d.vi->format.bitsPerSample < 8 || d.vi->format.bitsPerSample > 16) {
            vsapi->mapSetError(out, "ChromaticAberration: integer formats must be 8-16 bit");
            vsapi->freeNode(d.node);
            return;
        }
    } else if (d.vi->format.sampleType == stFloat) {
        if (d.vi->format.bitsPerSample != 16 && d.vi->format.bitsPerSample != 32) {
            vsapi->mapSetError(out, "ChromaticAberration: float formats must be 16-bit (half) or 32-bit");
            vsapi->freeNode(d.node);
            return;
        }
    }

    d.intensity = (float)vsapi->mapGetFloat(in, "intensity", 0, &err);
    if (err) d.intensity = 1.0f;

    if (d.intensity < 0.0f || d.intensity > 20.0f) {
        vsapi->mapSetError(out, "ChromaticAberration: intensity must be in range [0.0, 20.0]");
        vsapi->freeNode(d.node);
        return;
    }

    ChromaticAberrationData *data = (ChromaticAberrationData *)malloc(sizeof(d));
    *data = d;

    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "ChromaticAberration", d.vi, ca_get_frame, ca_free,
                            fmParallel, deps, 1, data, core);
}
