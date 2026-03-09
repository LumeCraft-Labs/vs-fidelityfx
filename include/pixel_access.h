#ifndef PIXEL_ACCESS_H
#define PIXEL_ACCESS_H

#include "VapourSynth4.h"
#include "VSHelper4.h"
#include "fsr1_common.h"
#include <stdint.h>
#include <math.h>

// Portable half-float <-> float conversion (IEEE 754)
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign; // +/- zero
        } else {
            // denorm -> normalize
            float val = ldexpf((float)mant, -24);
            uint32_t tmp;
            memcpy(&tmp, &val, sizeof(tmp));
            f = sign | tmp;
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | ((uint32_t)mant << 13); // inf/nan
    } else {
        f = sign | ((uint32_t)(exp - 15 + 127) << 23) | ((uint32_t)mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(result));
    return result;
}

static inline uint16_t float_to_half(float val) {
    uint32_t f;
    memcpy(&f, &val, sizeof(f));
    uint16_t sign = (uint16_t)((f >> 16) & 0x8000);
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return sign; // too small -> zero
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(1 - exp + 13);
        // round to nearest even
        uint32_t round_bit = 1u << (shift - 1);
        uint32_t result = mant >> shift;
        if ((mant & round_bit) && ((result & 1) || (mant & (round_bit - 1))))
            result++;
        return sign | (uint16_t)result;
    } else if (exp >= 31) {
        if (((f >> 23) & 0xFF) == 0xFF) {
            // inf or nan
            return sign | 0x7C00 | (uint16_t)(mant >> 13);
        }
        return sign | 0x7C00; // overflow -> inf
    }
    // round to nearest even
    uint16_t h = sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
    if ((mant >> 12) & 1) {
        if ((h & 1) || (mant & 0xFFF))
            h++;
    }
    return h;
}

// Pixel format tag — determined once at context init, avoids per-pixel format inspection
typedef enum {
    PF_U8,    // 8-bit unsigned integer
    PF_U16,   // 10-16 bit unsigned integer (stored as uint16_t)
    PF_F32,   // 32-bit float
    PF_F16    // 16-bit half float (stored as uint16_t)
} PixelFormatTag;

// Pixel loading context for callbacks
typedef struct {
    const VSFrame *frame;
    const VSVideoFormat *format;
    const uint8_t *plane_ptrs[3];  // R, G, B plane pointers (GRAY: all point to plane 0)
    ptrdiff_t strides[3];          // Byte strides
    ptrdiff_t elem_strides[3];     // Element strides (stride / sizeof(element))
    int width;
    int height;
    int numPlanes;                 // 1 for GRAY, 3 for RGB
    PixelFormatTag tag;            // Format tag for fast dispatch
    float norm_scale;              // 1.0/maxVal for int, 1.0 for float
} PixelLoadContext;

// Pixel store context — caches write pointers and format info
typedef struct {
    uint8_t *plane_ptrs[3];        // Write plane pointers
    ptrdiff_t strides[3];          // Byte strides
    ptrdiff_t elem_strides[3];     // Element strides
    int writePlanes;               // 1 for GRAY, 3 for RGB
    PixelFormatTag tag;
    float denorm_scale;            // maxVal for int, 1.0 for float
} PixelStoreContext;

// Initialize pixel loading context
void init_pixel_context(PixelLoadContext *ctx, const VSFrame *frame,
                       const VSAPI *vsapi);

// Initialize pixel store context
void init_store_context(PixelStoreContext *ctx, VSFrame *frame,
                       const VSAPI *vsapi);

// Clamp coordinate to valid range
static inline int clamp_coord(int coord, int max) {
    if (coord < 0) return 0;
    if (coord >= max) return max - 1;
    return coord;
}

// Load single pixel as float RGB [0.0, 1.0]
void load_pixel_rgb(float rgb[3], const PixelLoadContext *ctx, int x, int y);

// Store float RGB [0.0, 1.0] to pixel
void store_pixel_rgb(const PixelStoreContext *ctx, int x, int y, const float rgb[3]);

// Load 2x2 block of pixels for a single channel (gather4)
void gather4_channel(float result[4], const PixelLoadContext *ctx,
                    float x, float y, int channel);

// ========== PixelVec: Cross-architecture SIMD abstraction ==========
// Uses SSE on x86/x86_64 when available, scalar fallback otherwise

#ifdef __SSE2__
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2

typedef __m128 PixelVec;

static inline PixelVec pv_set(float r, float g, float b) {
    return _mm_set_ps(0.0f, b, g, r);
}
static inline PixelVec pv_set1(float x) { return _mm_set1_ps(x); }
static inline PixelVec pv_zero() { return _mm_setzero_ps(); }
static inline PixelVec pv_add(PixelVec a, PixelVec b) { return _mm_add_ps(a, b); }
static inline PixelVec pv_sub(PixelVec a, PixelVec b) { return _mm_sub_ps(a, b); }
static inline PixelVec pv_mul(PixelVec a, PixelVec b) { return _mm_mul_ps(a, b); }
static inline PixelVec pv_min(PixelVec a, PixelVec b) { return _mm_min_ps(a, b); }
static inline PixelVec pv_max(PixelVec a, PixelVec b) { return _mm_max_ps(a, b); }
static inline PixelVec pv_neg(PixelVec a) { return _mm_sub_ps(_mm_setzero_ps(), a); }

static inline float pv_extract(PixelVec v, int i) {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, v);
    return tmp[i];
}

static inline float pv_luma(PixelVec rgb) {
    __m128 weights = _mm_set_ps(0.0f, 0.5f, 1.0f, 0.5f);
    __m128 prod = _mm_mul_ps(rgb, weights);
    __m128 shuf1 = _mm_shuffle_ps(prod, prod, _MM_SHUFFLE(1, 0, 3, 2));
    __m128 sum1 = _mm_add_ps(prod, shuf1);
    __m128 shuf2 = _mm_shuffle_ps(sum1, sum1, _MM_SHUFFLE(0, 1, 0, 1));
    __m128 sum2 = _mm_add_ps(sum1, shuf2);
    return _mm_cvtss_f32(sum2);
}

// Newton-Raphson refined reciprocal: rcp = 2*rcp0 - rcp0*rcp0*x
static inline PixelVec pv_rcp_nr(PixelVec a) {
    __m128 rcp = _mm_rcp_ps(a);
    return _mm_sub_ps(_mm_add_ps(rcp, rcp), _mm_mul_ps(_mm_mul_ps(rcp, rcp), a));
}

#else // Scalar fallback

typedef struct { float v[4]; } PixelVec;

static inline PixelVec pv_set(float r, float g, float b) {
    PixelVec p; p.v[0] = r; p.v[1] = g; p.v[2] = b; p.v[3] = 0.0f; return p;
}
static inline PixelVec pv_set1(float x) {
    PixelVec p; p.v[0] = p.v[1] = p.v[2] = p.v[3] = x; return p;
}
static inline PixelVec pv_zero() {
    PixelVec p; p.v[0] = p.v[1] = p.v[2] = p.v[3] = 0.0f; return p;
}
static inline PixelVec pv_add(PixelVec a, PixelVec b) {
    PixelVec p; for (int i = 0; i < 4; i++) p.v[i] = a.v[i] + b.v[i]; return p;
}
static inline PixelVec pv_sub(PixelVec a, PixelVec b) {
    PixelVec p; for (int i = 0; i < 4; i++) p.v[i] = a.v[i] - b.v[i]; return p;
}
static inline PixelVec pv_mul(PixelVec a, PixelVec b) {
    PixelVec p; for (int i = 0; i < 4; i++) p.v[i] = a.v[i] * b.v[i]; return p;
}
static inline PixelVec pv_min(PixelVec a, PixelVec b) {
    PixelVec p; for (int i = 0; i < 4; i++) p.v[i] = a.v[i] < b.v[i] ? a.v[i] : b.v[i]; return p;
}
static inline PixelVec pv_max(PixelVec a, PixelVec b) {
    PixelVec p; for (int i = 0; i < 4; i++) p.v[i] = a.v[i] > b.v[i] ? a.v[i] : b.v[i]; return p;
}
static inline PixelVec pv_neg(PixelVec a) {
    PixelVec p; for (int i = 0; i < 4; i++) p.v[i] = -a.v[i]; return p;
}
static inline float pv_extract(PixelVec v, int i) { return v.v[i]; }

static inline float pv_luma(PixelVec rgb) {
    return rgb.v[0] * 0.5f + rgb.v[1] + rgb.v[2] * 0.5f;
}

static inline PixelVec pv_rcp_nr(PixelVec a) {
    PixelVec p;
    for (int i = 0; i < 4; i++) p.v[i] = a.v[i] != 0.0f ? 1.0f / a.v[i] : 0.0f;
    return p;
}

#endif // __SSE2__

// Common helpers (work with both paths)
static inline PixelVec pv_min3(PixelVec a, PixelVec b, PixelVec c) {
    return pv_min(pv_min(a, b), c);
}
static inline PixelVec pv_max3(PixelVec a, PixelVec b, PixelVec c) {
    return pv_max(pv_max(a, b), c);
}

// Load single pixel as PixelVec {R, G, B, 0}
PixelVec load_pixel_vec(const PixelLoadContext *ctx, int x, int y);

// Store PixelVec {R, G, B, 0} to pixel
void store_pixel_vec(const PixelStoreContext *ctx, int x, int y, PixelVec rgb);

// Bilinear-interpolated single-channel sample at float pixel coordinates
// channel: 0=R, 1=G, 2=B (for GRAY, all map to plane 0)
float sample_channel_bilinear(const PixelLoadContext *ctx, int channel, float px, float py);

// Read a single sample from one channel at integer coords (clamped)
float read_channel(const PixelLoadContext *ctx, int channel, int x, int y);

#endif // PIXEL_ACCESS_H
