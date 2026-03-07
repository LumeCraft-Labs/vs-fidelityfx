#include "../include/pixel_access.h"
#include <string.h>
#include <math.h>

void init_pixel_context(PixelLoadContext *ctx, const VSFrame *frame,
                       const VSAPI *vsapi) {
    ctx->frame = frame;
    ctx->format = vsapi->getVideoFrameFormat(frame);
    ctx->width = vsapi->getFrameWidth(frame, 0);
    ctx->height = vsapi->getFrameHeight(frame, 0);
    ctx->numPlanes = ctx->format->numPlanes;

    ctx->plane_ptrs[0] = vsapi->getReadPtr(frame, 0);
    ctx->strides[0] = vsapi->getStride(frame, 0);

    if (ctx->numPlanes >= 3) {
        // RGB: 0=R, 1=G, 2=B
        ctx->plane_ptrs[1] = vsapi->getReadPtr(frame, 1);
        ctx->strides[1] = vsapi->getStride(frame, 1);
        ctx->plane_ptrs[2] = vsapi->getReadPtr(frame, 2);
        ctx->strides[2] = vsapi->getStride(frame, 2);
    } else {
        // GRAY: all channels point to the single plane
        ctx->plane_ptrs[1] = ctx->plane_ptrs[0];
        ctx->strides[1] = ctx->strides[0];
        ctx->plane_ptrs[2] = ctx->plane_ptrs[0];
        ctx->strides[2] = ctx->strides[0];
    }
}

void load_pixel_rgb(float rgb[3], const PixelLoadContext *ctx, int x, int y) {
    // Clamp coordinates
    x = clamp_coord(x, ctx->width);
    y = clamp_coord(y, ctx->height);

    // Load based on format
    if (ctx->format->sampleType == stInteger) {
        int bits = ctx->format->bitsPerSample;
        if (bits == 8) {
            for (int c = 0; c < 3; c++) {
                const uint8_t *plane = ctx->plane_ptrs[c];
                rgb[c] = plane[y * ctx->strides[c] + x] / 255.0f;
            }
        } else {
            // 10/12/14/16-bit integer stored as uint16_t
            float scale = 1.0f / (float)((1 << bits) - 1);
            for (int c = 0; c < 3; c++) {
                const uint16_t *plane = (const uint16_t *)ctx->plane_ptrs[c];
                ptrdiff_t stride16 = ctx->strides[c] / sizeof(uint16_t);
                rgb[c] = plane[y * stride16 + x] * scale;
            }
        }
    } else if (ctx->format->sampleType == stFloat) {
        if (ctx->format->bitsPerSample == 32) {
            for (int c = 0; c < 3; c++) {
                const float *plane = (const float *)ctx->plane_ptrs[c];
                rgb[c] = plane[y * (ctx->strides[c] / sizeof(float)) + x];
            }
        } else {
            // FP16 half
            for (int c = 0; c < 3; c++) {
                const uint16_t *plane = (const uint16_t *)ctx->plane_ptrs[c];
                ptrdiff_t stride16 = ctx->strides[c] / sizeof(uint16_t);
                rgb[c] = half_to_float(plane[y * stride16 + x]);
            }
        }
    }
}

void store_pixel_rgb(VSFrame *frame, const VSAPI *vsapi,
                    int x, int y, const float rgb[3]) {
    const VSVideoFormat *format = vsapi->getVideoFrameFormat(frame);

    int numPlanes = format->numPlanes;

    int writePlanes = numPlanes >= 3 ? 3 : 1;

    if (format->sampleType == stInteger) {
        int bits = format->bitsPerSample;
        float maxVal = (float)((1 << bits) - 1);
        for (int c = 0; c < writePlanes; c++) {
            float clamped = fminf(fmaxf(rgb[c], 0.0f), 1.0f);
            if (bits == 8) {
                uint8_t *plane = vsapi->getWritePtr(frame, c);
                ptrdiff_t stride = vsapi->getStride(frame, c);
                plane[y * stride + x] = (uint8_t)(clamped * maxVal + 0.5f);
            } else {
                uint16_t *plane = (uint16_t *)vsapi->getWritePtr(frame, c);
                ptrdiff_t stride = vsapi->getStride(frame, c) / sizeof(uint16_t);
                plane[y * stride + x] = (uint16_t)(clamped * maxVal + 0.5f);
            }
        }
    } else if (format->sampleType == stFloat) {
        if (format->bitsPerSample == 32) {
            for (int c = 0; c < writePlanes; c++) {
                float *plane = (float *)vsapi->getWritePtr(frame, c);
                ptrdiff_t stride = vsapi->getStride(frame, c) / sizeof(float);
                plane[y * stride + x] = rgb[c];
            }
        } else {
            // FP16 half
            for (int c = 0; c < writePlanes; c++) {
                uint16_t *plane = (uint16_t *)vsapi->getWritePtr(frame, c);
                ptrdiff_t stride = vsapi->getStride(frame, c) / sizeof(uint16_t);
                plane[y * stride + x] = float_to_half(rgb[c]);
            }
        }
    }
}

void gather4_channel(float result[4], const PixelLoadContext *ctx,
                    float x, float y, int channel) {
    // Convert from normalized [0,1] texture coordinates to pixel coordinates
    float pixel_x = x * ctx->width;
    float pixel_y = y * ctx->height;

    // Get integer coordinates
    int ix = (int)floorf(pixel_x);
    int iy = (int)floorf(pixel_y);

    // Load 2x2 block with clamping
    int x0 = clamp_coord(ix, ctx->width);
    int x1 = clamp_coord(ix + 1, ctx->width);
    int y0 = clamp_coord(iy, ctx->height);
    int y1 = clamp_coord(iy + 1, ctx->height);

    const uint8_t *plane = ctx->plane_ptrs[channel];
    ptrdiff_t stride = ctx->strides[channel];

    // GPU gather4 ordering: [r, g, b, a] = [bottom-left, bottom-right, top-right, top-left]
    // For "ijfe" layout (e,f / i,j): [0]=i(bottom-left), [1]=j(bottom-right), [2]=f(top-right), [3]=e(top-left)
    // In image coords (Y-down): bottom=y1, top=y0, left=x0, right=x1
    if (ctx->format->sampleType == stInteger) {
        int bits = ctx->format->bitsPerSample;
        if (bits == 8) {
            float scale = 1.0f / 255.0f;
            result[0] = plane[y1 * stride + x0] * scale;
            result[1] = plane[y1 * stride + x1] * scale;
            result[2] = plane[y0 * stride + x1] * scale;
            result[3] = plane[y0 * stride + x0] * scale;
        } else {
            const uint16_t *p16 = (const uint16_t *)plane;
            ptrdiff_t s16 = stride / sizeof(uint16_t);
            float scale = 1.0f / (float)((1 << bits) - 1);
            result[0] = p16[y1 * s16 + x0] * scale;
            result[1] = p16[y1 * s16 + x1] * scale;
            result[2] = p16[y0 * s16 + x1] * scale;
            result[3] = p16[y0 * s16 + x0] * scale;
        }
    } else if (ctx->format->sampleType == stFloat) {
        if (ctx->format->bitsPerSample == 32) {
            const float *fp = (const float *)plane;
            ptrdiff_t sf = stride / sizeof(float);
            result[0] = fp[y1 * sf + x0];
            result[1] = fp[y1 * sf + x1];
            result[2] = fp[y0 * sf + x1];
            result[3] = fp[y0 * sf + x0];
        } else {
            const uint16_t *hp = (const uint16_t *)plane;
            ptrdiff_t sh = stride / sizeof(uint16_t);
            result[0] = half_to_float(hp[y1 * sh + x0]);
            result[1] = half_to_float(hp[y1 * sh + x1]);
            result[2] = half_to_float(hp[y0 * sh + x1]);
            result[3] = half_to_float(hp[y0 * sh + x0]);
        }
    }
}

PixelVec load_pixel_vec(const PixelLoadContext *ctx, int x, int y) {
    x = clamp_coord(x, ctx->width);
    y = clamp_coord(y, ctx->height);

    float r, g, b;
    if (ctx->format->sampleType == stInteger) {
        int bits = ctx->format->bitsPerSample;
        if (bits == 8) {
            float scale = 1.0f / 255.0f;
            r = ctx->plane_ptrs[0][y * ctx->strides[0] + x] * scale;
            g = ctx->plane_ptrs[1][y * ctx->strides[1] + x] * scale;
            b = ctx->plane_ptrs[2][y * ctx->strides[2] + x] * scale;
        } else {
            float scale = 1.0f / (float)((1 << bits) - 1);
            r = ((const uint16_t *)ctx->plane_ptrs[0])[y * (ctx->strides[0] / 2) + x] * scale;
            g = ((const uint16_t *)ctx->plane_ptrs[1])[y * (ctx->strides[1] / 2) + x] * scale;
            b = ((const uint16_t *)ctx->plane_ptrs[2])[y * (ctx->strides[2] / 2) + x] * scale;
        }
    } else if (ctx->format->bitsPerSample == 32) {
        r = ((const float *)ctx->plane_ptrs[0])[y * (ctx->strides[0] / 4) + x];
        g = ((const float *)ctx->plane_ptrs[1])[y * (ctx->strides[1] / 4) + x];
        b = ((const float *)ctx->plane_ptrs[2])[y * (ctx->strides[2] / 4) + x];
    } else {
        r = half_to_float(((const uint16_t *)ctx->plane_ptrs[0])[y * (ctx->strides[0] / 2) + x]);
        g = half_to_float(((const uint16_t *)ctx->plane_ptrs[1])[y * (ctx->strides[1] / 2) + x]);
        b = half_to_float(((const uint16_t *)ctx->plane_ptrs[2])[y * (ctx->strides[2] / 2) + x]);
    }
    return pv_set(r, g, b);
}

void store_pixel_vec(VSFrame *frame, const VSAPI *vsapi,
                    int x, int y, PixelVec rgb) {
    const VSVideoFormat *format = vsapi->getVideoFrameFormat(frame);
    int writePlanes = format->numPlanes >= 3 ? 3 : 1;

    float tmp[3];
    tmp[0] = pv_extract(rgb, 0);
    tmp[1] = pv_extract(rgb, 1);
    tmp[2] = pv_extract(rgb, 2);

    if (format->sampleType == stInteger) {
        int bits = format->bitsPerSample;
        float maxVal = (float)((1 << bits) - 1);
        for (int c = 0; c < writePlanes; c++) {
            float clamped = fminf(fmaxf(tmp[c], 0.0f), 1.0f);
            if (bits == 8) {
                uint8_t *plane = vsapi->getWritePtr(frame, c);
                plane[y * vsapi->getStride(frame, c) + x] = (uint8_t)(clamped * maxVal + 0.5f);
            } else {
                uint16_t *plane = (uint16_t *)vsapi->getWritePtr(frame, c);
                ptrdiff_t stride = vsapi->getStride(frame, c) / sizeof(uint16_t);
                plane[y * stride + x] = (uint16_t)(clamped * maxVal + 0.5f);
            }
        }
    } else if (format->bitsPerSample == 32) {
        for (int c = 0; c < writePlanes; c++) {
            float *plane = (float *)vsapi->getWritePtr(frame, c);
            ptrdiff_t stride = vsapi->getStride(frame, c) / sizeof(float);
            plane[y * stride + x] = tmp[c];
        }
    } else {
        for (int c = 0; c < writePlanes; c++) {
            uint16_t *plane = (uint16_t *)vsapi->getWritePtr(frame, c);
            ptrdiff_t stride = vsapi->getStride(frame, c) / sizeof(uint16_t);
            plane[y * stride + x] = float_to_half(tmp[c]);
        }
    }
}

float read_channel(const PixelLoadContext *ctx, int channel, int x, int y) {
    x = clamp_coord(x, ctx->width);
    y = clamp_coord(y, ctx->height);

    const uint8_t *plane = ctx->plane_ptrs[channel];
    ptrdiff_t stride = ctx->strides[channel];

    if (ctx->format->sampleType == stInteger) {
        int bits = ctx->format->bitsPerSample;
        if (bits == 8) {
            return plane[y * stride + x] / 255.0f;
        } else {
            float scale = 1.0f / (float)((1 << bits) - 1);
            const uint16_t *p16 = (const uint16_t *)plane;
            ptrdiff_t s16 = stride / sizeof(uint16_t);
            return p16[y * s16 + x] * scale;
        }
    } else if (ctx->format->bitsPerSample == 32) {
        const float *fp = (const float *)plane;
        ptrdiff_t sf = stride / sizeof(float);
        return fp[y * sf + x];
    } else {
        const uint16_t *hp = (const uint16_t *)plane;
        ptrdiff_t sh = stride / sizeof(uint16_t);
        return half_to_float(hp[y * sh + x]);
    }
}

float sample_channel_bilinear(const PixelLoadContext *ctx, int channel, float px, float py) {
    // px, py are in pixel coordinates (not normalized)
    float fx = floorf(px);
    float fy = floorf(py);
    float fracx = px - fx;
    float fracy = py - fy;

    int x0 = (int)fx;
    int y0 = (int)fy;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float s00 = read_channel(ctx, channel, x0, y0);
    float s10 = read_channel(ctx, channel, x1, y0);
    float s01 = read_channel(ctx, channel, x0, y1);
    float s11 = read_channel(ctx, channel, x1, y1);

    float top = s00 + (s10 - s00) * fracx;
    float bot = s01 + (s11 - s01) * fracx;
    return top + (bot - top) * fracy;
}
