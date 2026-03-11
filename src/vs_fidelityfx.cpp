#include "../include/vs_fidelityfx.h"
#include "VapourSynth4.h"
#include "VSHelper4.h"

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.hooke007.fidelityfx",
                         "fidelityfx",
                         "FidelityFX for VapourSynth",
                         VS_MAKE_VERSION(1, 1),
                         VAPOURSYNTH_API_VERSION,
                         0,
                         plugin);

    // EASU function
    vspapi->registerFunction("EASU",
                             "clip:vnode;width:int;height:int;fast:int:opt;",
                             "clip:vnode;",
                             easu_create, NULL, plugin);

    // RCAS function
    vspapi->registerFunction("RCAS",
                             "clip:vnode;sharpness:float:opt;",
                             "clip:vnode;",
                             rcas_create, NULL, plugin);

    // Chromatic Aberration function
    vspapi->registerFunction("CA",
                             "clip:vnode;intensity:float:opt;",
                             "clip:vnode;",
                             chromatic_aberration_create, NULL, plugin);

    // Vignette function
    vspapi->registerFunction("VIG",
                             "clip:vnode;intensity:float:opt;",
                             "clip:vnode;",
                             vignette_create, NULL, plugin);

    // Grain function
    vspapi->registerFunction("GRAIN",
                             "clip:vnode;scale:float:opt;amount:float:opt;seed:int:opt;",
                             "clip:vnode;",
                             grain_create, NULL, plugin);
}
