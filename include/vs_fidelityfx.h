#ifndef VS_FIDELITYFX_H
#define VS_FIDELITYFX_H

#include "VapourSynth4.h"
#include "VSHelper4.h"
#include "fsr1_common.h"

// RCAS filter data
typedef struct {
    VSNode *node;
    const VSVideoInfo *vi;
    float sharpness;
    FfxUInt32x4 constants;
} RcasData;

// EASU filter data
typedef struct {
    VSNode *node;
    VSVideoInfo vi;
    int input_width;
    int input_height;
    int fast;
    FfxUInt32x4 con0, con1, con2, con3;
} EasuData;

// RCAS functions
void rcas_create(const VSMap *in, VSMap *out, void *userData,
                VSCore *core, const VSAPI *vsapi);

// EASU functions
void easu_create(const VSMap *in, VSMap *out, void *userData,
                VSCore *core, const VSAPI *vsapi);

// Chromatic Aberration filter data
typedef struct {
    VSNode *node;
    const VSVideoInfo *vi;
    float intensity;
} ChromaticAberrationData;

void chromatic_aberration_create(const VSMap *in, VSMap *out, void *userData,
                                VSCore *core, const VSAPI *vsapi);

// Vignette filter data
typedef struct {
    VSNode *node;
    const VSVideoInfo *vi;
    float intensity;
} VignetteData;

void vignette_create(const VSMap *in, VSMap *out, void *userData,
                    VSCore *core, const VSAPI *vsapi);

// Film Grain filter data
typedef struct {
    VSNode *node;
    const VSVideoInfo *vi;
    float scale;
    float amount;
    int seed;  // -1 = use frame number
} GrainData;

void grain_create(const VSMap *in, VSMap *out, void *userData,
                 VSCore *core, const VSAPI *vsapi);

#endif // VS_FIDELITYFX_H
