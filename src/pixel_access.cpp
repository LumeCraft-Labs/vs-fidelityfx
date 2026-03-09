#include "../include/pixel_access.h"
#include <string.h>
#include <math.h>

// Helper: determine format tag from VSVideoFormat
static PixelFormatTag format_to_tag(const VSVideoFormat *fmt) {
    if (fmt->sampleType == stInteger) {
        return (fmt->bitsPerSample == 8) ? PF_U8 : PF_U16;
    } else {
        return (fmt->bitsPerSample == 32) ? PF_F32 : PF_F16;
    }
}

void init_pixel_context(PixelLoadContext *ctx, const VSFrame *frame,
                       const VSAPI *vsapi) {
    ctx->frame = frame;
    ctx->format = vsapi->getVideoFrameFormat(frame);
    ctx->width = vsapi->getFrameWidth(frame, 0);
    ctx->height = vsapi->getFrameHeight(frame, 0);
    ctx->numPlanes = ctx->format->numPlanes;
    ctx->tag = format_to_tag(ctx->format);

    // Precompute normalization scale
    if (ctx->tag == PF_U8) {
        ctx->norm_scale = 1.0f / 255.0f;
    } else if (ctx->tag == PF_U16) {
        ctx->norm_scale = 1.0f / (float)((1 << ctx->format->bitsPerSample) - 1);
    } else {
        ctx->norm_scale = 1.0f;
    }

    ctx->plane_ptrs[0] = vsapi->getReadPtr(frame, 0);
    ctx->strides[0] = vsapi->getStride(frame, 0);

    if (ctx->numPlanes >= 3) {
        ctx->plane_ptrs[1] = vsapi->getReadPtr(frame, 1);
        ctx->strides[1] = vsapi->getStride(frame, 1);
        ctx->plane_ptrs[2] = vsapi->getReadPtr(frame, 2);
        ctx->strides[2] = vsapi->getStride(frame, 2);
    } else {
        ctx->plane_ptrs[1] = ctx->plane_ptrs[0];
        ctx->strides[1] = ctx->strides[0];
        ctx->plane_ptrs[2] = ctx->plane_ptrs[0];
        ctx->strides[2] = ctx->strides[0];
    }

    // Precompute element strides
    for (int c = 0; c < 3; c++) {
        switch (ctx->tag) {
            case PF_U8:  ctx->elem_strides[c] = ctx->strides[c]; break;
            case PF_U16: ctx->elem_strides[c] = ctx->strides[c] / (ptrdiff_t)sizeof(uint16_t); break;
            case PF_F32: ctx->elem_strides[c] = ctx->strides[c] / (ptrdiff_t)sizeof(float); break;
            case PF_F16: ctx->elem_strides[c] = ctx->strides[c] / (ptrdiff_t)sizeof(uint16_t); break;
        }
    }
}

void init_store_context(PixelStoreContext *ctx, VSFrame *frame,
                       const VSAPI *vsapi) {
    const VSVideoFormat *fmt = vsapi->getVideoFrameFormat(frame);
    ctx->tag = format_to_tag(fmt);
    ctx->writePlanes = fmt->numPlanes >= 3 ? 3 : 1;

    if (ctx->tag == PF_U8) {
        ctx->denorm_scale = 255.0f;
    } else if (ctx->tag == PF_U16) {
        ctx->denorm_scale = (float)((1 << fmt->bitsPerSample) - 1);
    } else {
        ctx->denorm_scale = 1.0f;
    }

    for (int c = 0; c < ctx->writePlanes; c++) {
        ctx->plane_ptrs[c] = vsapi->getWritePtr(frame, c);
        ctx->strides[c] = vsapi->getStride(frame, c);
        switch (ctx->tag) {
            case PF_U8:  ctx->elem_strides[c] = ctx->strides[c]; break;
            case PF_U16: ctx->elem_strides[c] = ctx->strides[c] / (ptrdiff_t)sizeof(uint16_t); break;
            case PF_F32: ctx->elem_strides[c] = ctx->strides[c] / (ptrdiff_t)sizeof(float); break;
            case PF_F16: ctx->elem_strides[c] = ctx->strides[c] / (ptrdiff_t)sizeof(uint16_t); break;
        }
    }
}

void load_pixel_rgb(float rgb[3], const PixelLoadContext *ctx, int x, int y) {
    x = clamp_coord(x, ctx->width);
    y = clamp_coord(y, ctx->height);

    switch (ctx->tag) {
        case PF_U8: {
            float s = ctx->norm_scale;
            for (int c = 0; c < 3; c++)
                rgb[c] = ctx->plane_ptrs[c][y * ctx->elem_strides[c] + x] * s;
            break;
        }
        case PF_U16: {
            float s = ctx->norm_scale;
            for (int c = 0; c < 3; c++) {
                const uint16_t *p = (const uint16_t *)ctx->plane_ptrs[c];
                rgb[c] = p[y * ctx->elem_strides[c] + x] * s;
            }
            break;
        }
        case PF_F32:
            for (int c = 0; c < 3; c++) {
                const float *p = (const float *)ctx->plane_ptrs[c];
                rgb[c] = p[y * ctx->elem_strides[c] + x];
            }
            break;
        case PF_F16:
            for (int c = 0; c < 3; c++) {
                const uint16_t *p = (const uint16_t *)ctx->plane_ptrs[c];
                rgb[c] = half_to_float(p[y * ctx->elem_strides[c] + x]);
            }
            break;
    }
}

void store_pixel_rgb(const PixelStoreContext *ctx, int x, int y, const float rgb[3]) {
    switch (ctx->tag) {
        case PF_U8: {
            float s = ctx->denorm_scale;
            for (int c = 0; c < ctx->writePlanes; c++) {
                float clamped = fminf(fmaxf(rgb[c], 0.0f), 1.0f);
                ctx->plane_ptrs[c][y * ctx->elem_strides[c] + x] = (uint8_t)(clamped * s + 0.5f);
            }
            break;
        }
        case PF_U16: {
            float s = ctx->denorm_scale;
            for (int c = 0; c < ctx->writePlanes; c++) {
                float clamped = fminf(fmaxf(rgb[c], 0.0f), 1.0f);
                ((uint16_t *)ctx->plane_ptrs[c])[y * ctx->elem_strides[c] + x] = (uint16_t)(clamped * s + 0.5f);
            }
            break;
        }
        case PF_F32:
            for (int c = 0; c < ctx->writePlanes; c++)
                ((float *)ctx->plane_ptrs[c])[y * ctx->elem_strides[c] + x] = rgb[c];
            break;
        case PF_F16:
            for (int c = 0; c < ctx->writePlanes; c++)
                ((uint16_t *)ctx->plane_ptrs[c])[y * ctx->elem_strides[c] + x] = float_to_half(rgb[c]);
            break;
    }
}

void gather4_channel(float result[4], const PixelLoadContext *ctx,
                    float x, float y, int channel) {
    float pixel_x = x * ctx->width;
    float pixel_y = y * ctx->height;

    int ix = (int)floorf(pixel_x);
    int iy = (int)floorf(pixel_y);

    int x0 = clamp_coord(ix, ctx->width);
    int x1 = clamp_coord(ix + 1, ctx->width);
    int y0 = clamp_coord(iy, ctx->height);
    int y1 = clamp_coord(iy + 1, ctx->height);

    const uint8_t *plane = ctx->plane_ptrs[channel];
    ptrdiff_t es = ctx->elem_strides[channel];

    switch (ctx->tag) {
        case PF_U8: {
            float s = ctx->norm_scale;
            result[0] = plane[y1 * es + x0] * s;
            result[1] = plane[y1 * es + x1] * s;
            result[2] = plane[y0 * es + x1] * s;
            result[3] = plane[y0 * es + x0] * s;
            break;
        }
        case PF_U16: {
            const uint16_t *p = (const uint16_t *)plane;
            float s = ctx->norm_scale;
            result[0] = p[y1 * es + x0] * s;
            result[1] = p[y1 * es + x1] * s;
            result[2] = p[y0 * es + x1] * s;
            result[3] = p[y0 * es + x0] * s;
            break;
        }
        case PF_F32: {
            const float *p = (const float *)plane;
            result[0] = p[y1 * es + x0];
            result[1] = p[y1 * es + x1];
            result[2] = p[y0 * es + x1];
            result[3] = p[y0 * es + x0];
            break;
        }
        case PF_F16: {
            const uint16_t *p = (const uint16_t *)plane;
            result[0] = half_to_float(p[y1 * es + x0]);
            result[1] = half_to_float(p[y1 * es + x1]);
            result[2] = half_to_float(p[y0 * es + x1]);
            result[3] = half_to_float(p[y0 * es + x0]);
            break;
        }
    }
}

PixelVec load_pixel_vec(const PixelLoadContext *ctx, int x, int y) {
    x = clamp_coord(x, ctx->width);
    y = clamp_coord(y, ctx->height);

    float r, g, b;
    switch (ctx->tag) {
        case PF_U8: {
            float s = ctx->norm_scale;
            r = ctx->plane_ptrs[0][y * ctx->elem_strides[0] + x] * s;
            g = ctx->plane_ptrs[1][y * ctx->elem_strides[1] + x] * s;
            b = ctx->plane_ptrs[2][y * ctx->elem_strides[2] + x] * s;
            break;
        }
        case PF_U16: {
            float s = ctx->norm_scale;
            r = ((const uint16_t *)ctx->plane_ptrs[0])[y * ctx->elem_strides[0] + x] * s;
            g = ((const uint16_t *)ctx->plane_ptrs[1])[y * ctx->elem_strides[1] + x] * s;
            b = ((const uint16_t *)ctx->plane_ptrs[2])[y * ctx->elem_strides[2] + x] * s;
            break;
        }
        case PF_F32:
            r = ((const float *)ctx->plane_ptrs[0])[y * ctx->elem_strides[0] + x];
            g = ((const float *)ctx->plane_ptrs[1])[y * ctx->elem_strides[1] + x];
            b = ((const float *)ctx->plane_ptrs[2])[y * ctx->elem_strides[2] + x];
            break;
        case PF_F16:
            r = half_to_float(((const uint16_t *)ctx->plane_ptrs[0])[y * ctx->elem_strides[0] + x]);
            g = half_to_float(((const uint16_t *)ctx->plane_ptrs[1])[y * ctx->elem_strides[1] + x]);
            b = half_to_float(((const uint16_t *)ctx->plane_ptrs[2])[y * ctx->elem_strides[2] + x]);
            break;
    }
    return pv_set(r, g, b);
}

void store_pixel_vec(const PixelStoreContext *ctx, int x, int y, PixelVec rgb) {
    float tmp[3];
    tmp[0] = pv_extract(rgb, 0);
    tmp[1] = pv_extract(rgb, 1);
    tmp[2] = pv_extract(rgb, 2);

    switch (ctx->tag) {
        case PF_U8: {
            float s = ctx->denorm_scale;
            for (int c = 0; c < ctx->writePlanes; c++) {
                float clamped = fminf(fmaxf(tmp[c], 0.0f), 1.0f);
                ctx->plane_ptrs[c][y * ctx->elem_strides[c] + x] = (uint8_t)(clamped * s + 0.5f);
            }
            break;
        }
        case PF_U16: {
            float s = ctx->denorm_scale;
            for (int c = 0; c < ctx->writePlanes; c++) {
                float clamped = fminf(fmaxf(tmp[c], 0.0f), 1.0f);
                ((uint16_t *)ctx->plane_ptrs[c])[y * ctx->elem_strides[c] + x] = (uint16_t)(clamped * s + 0.5f);
            }
            break;
        }
        case PF_F32:
            for (int c = 0; c < ctx->writePlanes; c++)
                ((float *)ctx->plane_ptrs[c])[y * ctx->elem_strides[c] + x] = tmp[c];
            break;
        case PF_F16:
            for (int c = 0; c < ctx->writePlanes; c++)
                ((uint16_t *)ctx->plane_ptrs[c])[y * ctx->elem_strides[c] + x] = float_to_half(tmp[c]);
            break;
    }
}

float read_channel(const PixelLoadContext *ctx, int channel, int x, int y) {
    x = clamp_coord(x, ctx->width);
    y = clamp_coord(y, ctx->height);

    const uint8_t *plane = ctx->plane_ptrs[channel];
    ptrdiff_t es = ctx->elem_strides[channel];

    switch (ctx->tag) {
        case PF_U8:
            return plane[y * es + x] * ctx->norm_scale;
        case PF_U16:
            return ((const uint16_t *)plane)[y * es + x] * ctx->norm_scale;
        case PF_F32:
            return ((const float *)plane)[y * es + x];
        case PF_F16:
            return half_to_float(((const uint16_t *)plane)[y * es + x]);
    }
    return 0.0f;
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
