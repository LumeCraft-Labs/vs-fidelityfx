#ifndef FSR1_COMMON_H
#define FSR1_COMMON_H

#include <stdint.h>
#include <math.h>

// Define FFX_CPU to use CPU implementations
#define FFX_CPU

// Include FidelityFX headers
#include "../fidelityfx/ffx_common_types.h"
#include "../fidelityfx/ffx_core_cpu.h"

// Missing FidelityFX helper functions
static inline FfxFloat32 ffxMax3(FfxFloat32 a, FfxFloat32 b, FfxFloat32 c) {
    return ffxMax(ffxMax(a, b), c);
}

static inline FfxFloat32 ffxMin3(FfxFloat32 a, FfxFloat32 b, FfxFloat32 c) {
    return ffxMin(ffxMin(a, b), c);
}

static inline FfxFloat32 ffxAbs(FfxFloat32 x) {
    return fabsf(x);
}

static inline FfxFloat32 ffxExp2(FfxFloat32 x) {
    return exp2f(x);
}

static inline FfxFloat32 ffxApproximateReciprocalMedium(FfxFloat32 x) {
    return ffxReciprocal(x);
}

// NaN-safe versions of min/max/saturate for CPU.
// GPU saturate(NaN)=0, but CPU ffxMin/ffxMax propagate NaN since NaN comparisons are always false.
static inline FfxFloat32 ffxSaturateSafe(FfxFloat32 x) {
    return fminf(1.0f, fmaxf(0.0f, x));
}

// ffxAsFloat - convert uint32 bits to float
static inline FfxFloat32 ffxAsFloat(FfxUInt32 x) {
    union { FfxUInt32 u; FfxFloat32 f; } bits;
    bits.u = x;
    return bits.f;
}

// FP16 versions
#if FFX_HALF
static inline FfxFloat16 ffxMax3Half(FfxFloat16 a, FfxFloat16 b, FfxFloat16 c) {
    return ffxMax(ffxMax(a, b), c);
}

static inline FfxFloat16 ffxMin3Half(FfxFloat16 a, FfxFloat16 b, FfxFloat16 c) {
    return ffxMin(ffxMin(a, b), c);
}

static inline FfxFloat16 ffxApproximateReciprocalMediumHalf(FfxFloat16 x) {
    return ffxReciprocalHalf(x);
}

static inline FfxFloat16 ffxApproximateReciprocalSquareRootHalf(FfxFloat16 x) {
    return ffxRsqrtHalf(x);
}
#endif

#endif // FSR1_COMMON_H
