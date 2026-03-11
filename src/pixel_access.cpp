#include "../include/pixel_access.h"
#include <string.h>
#include <math.h>

static PixelFormat detect_pixfmt(const VSVideoFormat *format) {
    if (format->sampleType == stInteger)
        return format->bitsPerSample == 8 ? PF_INT8 : PF_INT16;
    else
        return format->bitsPerSample == 32 ? PF_FLOAT32 : PF_FLOAT16;
}

static int pixfmt_elem_size(PixelFormat pf) {
    switch (pf) {
        case PF_INT8:    return 1;
        case PF_FLOAT32: return 4;
        default:         return 2; // PF_INT16, PF_FLOAT16
    }
}

void init_pixel_context(PixelLoadContext *ctx, const VSFrame *frame,
                       const VSAPI *vsapi) {
    const VSVideoFormat *format = vsapi->getVideoFrameFormat(frame);
    ctx->width = vsapi->getFrameWidth(frame, 0);
    ctx->height = vsapi->getFrameHeight(frame, 0);
    ctx->numPlanes = format->numPlanes;
    ctx->pixfmt = detect_pixfmt(format);

    if (format->sampleType == stInteger)
        ctx->loadScale = 1.0f / (float)((1 << format->bitsPerSample) - 1);
    else
        ctx->loadScale = 1.0f;

    int elem = pixfmt_elem_size(ctx->pixfmt);

    ctx->plane_ptrs[0] = vsapi->getReadPtr(frame, 0);
    ctx->strides[0] = vsapi->getStride(frame, 0) / elem;

    if (ctx->numPlanes >= 3) {
        ctx->plane_ptrs[1] = vsapi->getReadPtr(frame, 1);
        ctx->strides[1] = vsapi->getStride(frame, 1) / elem;
        ctx->plane_ptrs[2] = vsapi->getReadPtr(frame, 2);
        ctx->strides[2] = vsapi->getStride(frame, 2) / elem;
    } else {
        ctx->plane_ptrs[1] = ctx->plane_ptrs[0];
        ctx->strides[1] = ctx->strides[0];
        ctx->plane_ptrs[2] = ctx->plane_ptrs[0];
        ctx->strides[2] = ctx->strides[0];
    }
}

// Forward declarations of format-specialized store functions
static void store_rgb_int8(const PixelStoreContext *ctx, int x, int y, const float rgb[3]);
static void store_rgb_int16(const PixelStoreContext *ctx, int x, int y, const float rgb[3]);
static void store_rgb_f32(const PixelStoreContext *ctx, int x, int y, const float rgb[3]);
static void store_rgb_f16(const PixelStoreContext *ctx, int x, int y, const float rgb[3]);
static void store_vec_int8(const PixelStoreContext *ctx, int x, int y, PixelVec rgb);
static void store_vec_int16(const PixelStoreContext *ctx, int x, int y, PixelVec rgb);
static void store_vec_f32(const PixelStoreContext *ctx, int x, int y, PixelVec rgb);
static void store_vec_f16(const PixelStoreContext *ctx, int x, int y, PixelVec rgb);

void init_store_context(PixelStoreContext *ctx, VSFrame *frame,
                       const VSAPI *vsapi) {
    const VSVideoFormat *format = vsapi->getVideoFrameFormat(frame);
    ctx->writePlanes = format->numPlanes >= 3 ? 3 : 1;
    ctx->pixfmt = detect_pixfmt(format);

    if (format->sampleType == stInteger)
        ctx->storeScale = (float)((1 << format->bitsPerSample) - 1);
    else
        ctx->storeScale = 1.0f;

    int elem = pixfmt_elem_size(ctx->pixfmt);

    for (int c = 0; c < ctx->writePlanes; c++) {
        ctx->plane_ptrs[c] = vsapi->getWritePtr(frame, c);
        ctx->strides[c] = vsapi->getStride(frame, c) / elem;
    }

    // format-specialized function pointers
    static const StoreVecFn vec_table[] = { store_vec_int8, store_vec_int16, store_vec_f32, store_vec_f16 };
    static const StoreRgbFn rgb_table[] = { store_rgb_int8, store_rgb_int16, store_rgb_f32, store_rgb_f16 };
    ctx->store_vec = vec_table[ctx->pixfmt];
    ctx->store_rgb = rgb_table[ctx->pixfmt];
}

void load_pixel_rgb(float rgb[3], const PixelLoadContext *ctx, int x, int y) {
    x = clamp_coord(x, ctx->width);
    y = clamp_coord(y, ctx->height);

    switch (ctx->pixfmt) {
        case PF_INT8: {
            float scale = ctx->loadScale;
            for (int c = 0; c < 3; c++)
                rgb[c] = ctx->plane_ptrs[c][y * ctx->strides[c] + x] * scale;
            break;
        }
        case PF_INT16: {
            float scale = ctx->loadScale;
            for (int c = 0; c < 3; c++)
                rgb[c] = ((const uint16_t *)ctx->plane_ptrs[c])[y * ctx->strides[c] + x] * scale;
            break;
        }
        case PF_FLOAT32:
            for (int c = 0; c < 3; c++)
                rgb[c] = ((const float *)ctx->plane_ptrs[c])[y * ctx->strides[c] + x];
            break;
        case PF_FLOAT16:
            for (int c = 0; c < 3; c++)
                rgb[c] = half_to_float(((const uint16_t *)ctx->plane_ptrs[c])[y * ctx->strides[c] + x]);
            break;
    }
}

// ================== Format-specialized store functions =======================

static void store_rgb_int8(const PixelStoreContext *ctx, int x, int y, const float rgb[3]) {
    float maxVal = ctx->storeScale;
    for (int c = 0; c < ctx->writePlanes; c++) {
        float clamped = fminf(fmaxf(rgb[c], 0.0f), 1.0f);
        ctx->plane_ptrs[c][y * ctx->strides[c] + x] = (uint8_t)(clamped * maxVal + 0.5f);
    }
}
static void store_rgb_int16(const PixelStoreContext *ctx, int x, int y, const float rgb[3]) {
    float maxVal = ctx->storeScale;
    for (int c = 0; c < ctx->writePlanes; c++) {
        float clamped = fminf(fmaxf(rgb[c], 0.0f), 1.0f);
        ((uint16_t *)ctx->plane_ptrs[c])[y * ctx->strides[c] + x] = (uint16_t)(clamped * maxVal + 0.5f);
    }
}
static void store_rgb_f32(const PixelStoreContext *ctx, int x, int y, const float rgb[3]) {
    for (int c = 0; c < ctx->writePlanes; c++)
        ((float *)ctx->plane_ptrs[c])[y * ctx->strides[c] + x] = rgb[c];
}
static void store_rgb_f16(const PixelStoreContext *ctx, int x, int y, const float rgb[3]) {
    for (int c = 0; c < ctx->writePlanes; c++)
        ((uint16_t *)ctx->plane_ptrs[c])[y * ctx->strides[c] + x] = float_to_half(rgb[c]);
}

void store_pixel_rgb(const PixelStoreContext *ctx, int x, int y, const float rgb[3]) {
    ctx->store_rgb(ctx, x, y, rgb);
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
    ptrdiff_t stride = ctx->strides[channel];

    // GPU gather4 ordering (Y-down): [0]=bottom-left, [1]=bottom-right, [2]=top-right, [3]=top-left
    switch (ctx->pixfmt) {
        case PF_INT8: {
            float scale = ctx->loadScale;
            result[0] = plane[y1 * stride + x0] * scale;
            result[1] = plane[y1 * stride + x1] * scale;
            result[2] = plane[y0 * stride + x1] * scale;
            result[3] = plane[y0 * stride + x0] * scale;
            break;
        }
        case PF_INT16: {
            const uint16_t *p16 = (const uint16_t *)plane;
            float scale = ctx->loadScale;
            result[0] = p16[y1 * stride + x0] * scale;
            result[1] = p16[y1 * stride + x1] * scale;
            result[2] = p16[y0 * stride + x1] * scale;
            result[3] = p16[y0 * stride + x0] * scale;
            break;
        }
        case PF_FLOAT32: {
            const float *fp = (const float *)plane;
            result[0] = fp[y1 * stride + x0];
            result[1] = fp[y1 * stride + x1];
            result[2] = fp[y0 * stride + x1];
            result[3] = fp[y0 * stride + x0];
            break;
        }
        case PF_FLOAT16: {
            const uint16_t *hp = (const uint16_t *)plane;
            result[0] = half_to_float(hp[y1 * stride + x0]);
            result[1] = half_to_float(hp[y1 * stride + x1]);
            result[2] = half_to_float(hp[y0 * stride + x1]);
            result[3] = half_to_float(hp[y0 * stride + x0]);
            break;
        }
    }
}

PixelVec load_pixel_vec(const PixelLoadContext *ctx, int x, int y) {
    x = clamp_coord(x, ctx->width);
    y = clamp_coord(y, ctx->height);

    float r = 0.0f, g = 0.0f, b = 0.0f;
    switch (ctx->pixfmt) {
        case PF_INT8: {
            float scale = ctx->loadScale;
            r = ctx->plane_ptrs[0][y * ctx->strides[0] + x] * scale;
            g = ctx->plane_ptrs[1][y * ctx->strides[1] + x] * scale;
            b = ctx->plane_ptrs[2][y * ctx->strides[2] + x] * scale;
            break;
        }
        case PF_INT16: {
            float scale = ctx->loadScale;
            r = ((const uint16_t *)ctx->plane_ptrs[0])[y * ctx->strides[0] + x] * scale;
            g = ((const uint16_t *)ctx->plane_ptrs[1])[y * ctx->strides[1] + x] * scale;
            b = ((const uint16_t *)ctx->plane_ptrs[2])[y * ctx->strides[2] + x] * scale;
            break;
        }
        case PF_FLOAT32:
            r = ((const float *)ctx->plane_ptrs[0])[y * ctx->strides[0] + x];
            g = ((const float *)ctx->plane_ptrs[1])[y * ctx->strides[1] + x];
            b = ((const float *)ctx->plane_ptrs[2])[y * ctx->strides[2] + x];
            break;
        case PF_FLOAT16:
            r = half_to_float(((const uint16_t *)ctx->plane_ptrs[0])[y * ctx->strides[0] + x]);
            g = half_to_float(((const uint16_t *)ctx->plane_ptrs[1])[y * ctx->strides[1] + x]);
            b = half_to_float(((const uint16_t *)ctx->plane_ptrs[2])[y * ctx->strides[2] + x]);
            break;
    }
    return pv_set(r, g, b);
}

static void store_vec_int8(const PixelStoreContext *ctx, int x, int y, PixelVec rgb) {
    float maxVal = ctx->storeScale;
    for (int c = 0; c < ctx->writePlanes; c++) {
        float clamped = fminf(fmaxf(pv_extract(rgb, c), 0.0f), 1.0f);
        ctx->plane_ptrs[c][y * ctx->strides[c] + x] = (uint8_t)(clamped * maxVal + 0.5f);
    }
}
static void store_vec_int16(const PixelStoreContext *ctx, int x, int y, PixelVec rgb) {
    float maxVal = ctx->storeScale;
    for (int c = 0; c < ctx->writePlanes; c++) {
        float clamped = fminf(fmaxf(pv_extract(rgb, c), 0.0f), 1.0f);
        ((uint16_t *)ctx->plane_ptrs[c])[y * ctx->strides[c] + x] = (uint16_t)(clamped * maxVal + 0.5f);
    }
}
static void store_vec_f32(const PixelStoreContext *ctx, int x, int y, PixelVec rgb) {
    for (int c = 0; c < ctx->writePlanes; c++)
        ((float *)ctx->plane_ptrs[c])[y * ctx->strides[c] + x] = pv_extract(rgb, c);
}
static void store_vec_f16(const PixelStoreContext *ctx, int x, int y, PixelVec rgb) {
    for (int c = 0; c < ctx->writePlanes; c++)
        ((uint16_t *)ctx->plane_ptrs[c])[y * ctx->strides[c] + x] = float_to_half(pv_extract(rgb, c));
}

void store_pixel_vec(const PixelStoreContext *ctx, int x, int y, PixelVec rgb) {
    ctx->store_vec(ctx, x, y, rgb);
}

float read_channel(const PixelLoadContext *ctx, int channel, int x, int y) {
    x = clamp_coord(x, ctx->width);
    y = clamp_coord(y, ctx->height);

    const uint8_t *plane = ctx->plane_ptrs[channel];
    ptrdiff_t stride = ctx->strides[channel];

    switch (ctx->pixfmt) {
        case PF_INT8:
            return plane[y * stride + x] * ctx->loadScale;
        case PF_INT16:
            return ((const uint16_t *)plane)[y * stride + x] * ctx->loadScale;
        case PF_FLOAT32:
            return ((const float *)plane)[y * stride + x];
        case PF_FLOAT16:
            return half_to_float(((const uint16_t *)plane)[y * stride + x]);
        default:
            return 0.0f;
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

float *convert_to_float_planes(float *planes[3], int *out_stride,
                               const PixelLoadContext *ctx) {
    int w = ctx->width;
    int h = ctx->height;
    int nPlanes = ctx->numPlanes >= 3 ? 3 : 1;

    float *buf = (float *)malloc((size_t)w * h * nPlanes * sizeof(float));
    if (!buf) return NULL;

    planes[0] = buf;
    if (nPlanes >= 3) {
        planes[1] = buf + (size_t)w * h;
        planes[2] = buf + (size_t)w * h * 2;
    } else {
        planes[1] = planes[0];
        planes[2] = planes[0];
    }
    *out_stride = w;

    for (int c = 0; c < nPlanes; c++) {
        float *dst = planes[c];
        const uint8_t *src_plane = ctx->plane_ptrs[c];
        ptrdiff_t src_stride = ctx->strides[c];

        switch (ctx->pixfmt) {
            case PF_INT8: {
                float scale = ctx->loadScale;
                for (int y = 0; y < h; y++) {
                    const uint8_t *row = src_plane + y * src_stride;
                    float *out = dst + y * w;
                    for (int x = 0; x < w; x++)
                        out[x] = row[x] * scale;
                }
                break;
            }
            case PF_INT16: {
                float scale = ctx->loadScale;
                for (int y = 0; y < h; y++) {
                    const uint16_t *row = (const uint16_t *)src_plane + y * src_stride;
                    float *out = dst + y * w;
                    for (int x = 0; x < w; x++)
                        out[x] = row[x] * scale;
                }
                break;
            }
            case PF_FLOAT32: {
                for (int y = 0; y < h; y++) {
                    const float *row = (const float *)src_plane + y * src_stride;
                    float *out = dst + y * w;
                    memcpy(out, row, w * sizeof(float));
                }
                break;
            }
            case PF_FLOAT16: {
                for (int y = 0; y < h; y++) {
                    const uint16_t *row = (const uint16_t *)src_plane + y * src_stride;
                    float *out = dst + y * w;
                    for (int x = 0; x < w; x++)
                        out[x] = half_to_float(row[x]);
                }
                break;
            }
        }
    }

    return buf;
}

float *convert_to_rgbl(int *out_stride, const PixelLoadContext *ctx) {
    int w = ctx->width;
    int h = ctx->height;
    // 4 floats per pixel (R, G, B, Luma), 16-byte aligned
#ifdef _WIN32
    float *buf = (float *)_aligned_malloc((size_t)w * h * 4 * sizeof(float), 16);
#else
    float *buf = (float *)aligned_alloc(16, (size_t)w * h * 4 * sizeof(float));
#endif
    if (!buf) return NULL;
    *out_stride = w;

    const uint8_t *sp0 = ctx->plane_ptrs[0];
    const uint8_t *sp1 = ctx->plane_ptrs[1];
    const uint8_t *sp2 = ctx->plane_ptrs[2];
    ptrdiff_t st0 = ctx->strides[0], st1 = ctx->strides[1], st2 = ctx->strides[2];

    for (int y = 0; y < h; y++) {
        float *dst = buf + (size_t)y * w * 4;
        switch (ctx->pixfmt) {
            case PF_INT8: {
                float s = ctx->loadScale;
                const uint8_t *r = sp0 + y * st0;
                const uint8_t *g = sp1 + y * st1;
                const uint8_t *b = sp2 + y * st2;
                for (int x = 0; x < w; x++) {
                    float rf = r[x] * s, gf = g[x] * s, bf = b[x] * s;
                    dst[0] = rf; dst[1] = gf; dst[2] = bf;
                    dst[3] = rf * 0.5f + gf + bf * 0.5f;
                    dst += 4;
                }
                break;
            }
            case PF_INT16: {
                float s = ctx->loadScale;
                const uint16_t *r = (const uint16_t *)sp0 + y * st0;
                const uint16_t *g = (const uint16_t *)sp1 + y * st1;
                const uint16_t *b = (const uint16_t *)sp2 + y * st2;
                for (int x = 0; x < w; x++) {
                    float rf = r[x] * s, gf = g[x] * s, bf = b[x] * s;
                    dst[0] = rf; dst[1] = gf; dst[2] = bf;
                    dst[3] = rf * 0.5f + gf + bf * 0.5f;
                    dst += 4;
                }
                break;
            }
            case PF_FLOAT32: {
                const float *r = (const float *)sp0 + y * st0;
                const float *g = (const float *)sp1 + y * st1;
                const float *b = (const float *)sp2 + y * st2;
                for (int x = 0; x < w; x++) {
                    float rf = r[x], gf = g[x], bf = b[x];
                    dst[0] = rf; dst[1] = gf; dst[2] = bf;
                    dst[3] = rf * 0.5f + gf + bf * 0.5f;
                    dst += 4;
                }
                break;
            }
            case PF_FLOAT16: {
                const uint16_t *r = (const uint16_t *)sp0 + y * st0;
                const uint16_t *g = (const uint16_t *)sp1 + y * st1;
                const uint16_t *b = (const uint16_t *)sp2 + y * st2;
                for (int x = 0; x < w; x++) {
                    float rf = half_to_float(r[x]);
                    float gf = half_to_float(g[x]);
                    float bf = half_to_float(b[x]);
                    dst[0] = rf; dst[1] = gf; dst[2] = bf;
                    dst[3] = rf * 0.5f + gf + bf * 0.5f;
                    dst += 4;
                }
                break;
            }
        }
    }
    return buf;
}
