// Styx2Upscaler - ReShade add-on for Styx: Shards of Darkness (UE 4.13, DX11).
//
// Replaces the game's temporal anti-aliasing (TAA) pass with FSR 4 / DLSS / XeSS through OptiScaler.
// UE4 already renders with sub-pixel jitter and writes a velocity buffer for moving objects, so the add-on
// only has to recognise the TAA pass, read the camera from the View uniform buffer, build full motion
// vectors and hand everything to the upscaler. The upscaled image is written into the TAA's render target;
// bloom, tone mapping and UI follow as usual. See NOTES.md for the renderer details.

#define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#define ImTextureID ImU64
#include <imgui.h>  // Dear ImGui 1.92.5 headers (MIT): ReShade provides the implementation for overlays
#include <reshade.hpp>
#include <d3d11_1.h>
#include <wincodec.h>
#include <nvsdk_ngx.h>  // types and parameter names only: the functions come from OptiScaler, see LoadNgx
#include "ffx_antilag2_dx11.h"  // AMD Anti-Lag 2 SDK (MIT), header-only
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "motion_cs.h"
#include "sanitize_cs.h"
#include "rcas_cs.h"

using namespace reshade::api;

extern "C" __declspec(dllexport) const char* NAME = "Styx 2 Upscaler";
extern "C" __declspec(dllexport) const char* DESCRIPTION =
    "FSR 4 / DLSS / XeSS instead of TAA for Styx: Shards of Darkness (DX11) through OptiScaler.";

static void Log(reshade::log::level level, const char* fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    reshade::log::message(level, buf);
}
#define LOGI(...) Log(reshade::log::level::info, __VA_ARGS__)
#define LOGW(...) Log(reshade::log::level::warning, __VA_ARGS__)

static bool g_enabled = true;  // overlay: upscaler on / off (off = the game's own TAA)
static int g_logCandidates = 0;  // [Debug] LogCandidates: frames for which every TAA-like draw is logged
static bool g_logActive = false;  // logging candidates in the current frame

// Pixel shader -> FNV-1a hash of its bytecode: identifies shaders across launches (handles change).
static std::mutex g_psMutex;
static std::unordered_map<uint64_t, uint64_t> g_psHash;

static void OnInitPipeline(device*, pipeline_layout, uint32_t count, const pipeline_subobject* subs, pipeline pipe)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (subs[i].type != pipeline_subobject_type::pixel_shader || !subs[i].data)
            continue;
        const auto* d = static_cast<const shader_desc*>(subs[i].data);
        uint64_t h = 1469598103934665603ull;
        for (size_t b = 0; b < d->code_size; ++b)
            h = (h ^ static_cast<const uint8_t*>(d->code)[b]) * 1099511628211ull;
        std::lock_guard<std::mutex> lock(g_psMutex);
        g_psHash[pipe.handle] = h;
    }
}

static uint64_t g_ps = 0;

static uint64_t PsHash(uint64_t ps)
{
    std::lock_guard<std::mutex> lock(g_psMutex);
    auto it = g_psHash.find(ps);
    return it != g_psHash.end() ? it->second : 0;
}
static int g_debugView = 0;    // overlay: 0 off, 1 motion vectors

// ---------------------------------------------------------------- view info cache

struct ViewInfo {
    uint64_t res = 0;
    uint32_t width = 0, height = 0;
    format fmt = format::unknown;
};
static std::unordered_map<uint64_t, ViewInfo> g_viewInfo;
static std::mutex g_viewMutex;

static ViewInfo GetViewInfo(device* dev, uint64_t view)
{
    std::lock_guard<std::mutex> lock(g_viewMutex);
    auto it = g_viewInfo.find(view);
    if (it != g_viewInfo.end())
        return it->second;
    ViewInfo vi;
    resource r = dev->get_resource_from_view(resource_view{view});
    vi.res = r.handle;
    if (r.handle) {
        resource_desc d = dev->get_resource_desc(r);
        vi.width = d.texture.width;
        vi.height = d.texture.height;
        vi.fmt = d.texture.format;
    }
    return g_viewInfo.emplace(view, vi).first->second;
}

static void OnDestroyResourceView(device*, resource_view view)
{
    std::lock_guard<std::mutex> lock(g_viewMutex);
    g_viewInfo.erase(view.handle);
}

// ---------------------------------------------------------------- View uniform buffer capture
// UE4 uploads its per-view constants (FViewUniformShaderParameters, 4096 bytes in this build) as a uniform
// buffer; we keep a CPU copy of every buffer of that size so the TAA draw can read the camera and jitter.

static const uint32_t kViewSize = 4096;
static std::mutex g_cbMutex;
static std::unordered_map<uint64_t, uint32_t> g_cbSize;               // buffer -> size (0 = not tracked)
static std::unordered_map<uint64_t, std::vector<uint8_t>> g_viewData;  // buffer -> last written contents
static std::unordered_map<uint64_t, void*> g_mapped;

static bool IsViewSized(device* dev, resource res)
{
    auto it = g_cbSize.find(res.handle);
    if (it != g_cbSize.end())
        return it->second == kViewSize;
    resource_desc d = dev->get_resource_desc(res);
    const bool cb = d.type == resource_type::buffer &&
                    (d.usage & resource_usage::constant_buffer) == resource_usage::constant_buffer;
    g_cbSize.emplace(res.handle, cb ? static_cast<uint32_t>(d.buffer.size) : 0);
    return cb && d.buffer.size == kViewSize;
}

static void OnInitResource(device* dev, const resource_desc& desc, const subresource_data* initial, resource_usage, resource res)
{
    if (desc.type != resource_type::buffer || desc.buffer.size != kViewSize || !initial || !initial->data)
        return;
    std::lock_guard<std::mutex> lock(g_cbMutex);
    if (!IsViewSized(dev, res))
        return;
    auto& v = g_viewData[res.handle];
    v.assign(static_cast<const uint8_t*>(initial->data), static_cast<const uint8_t*>(initial->data) + kViewSize);
}

static void OnDestroyResource(device*, resource res)
{
    std::lock_guard<std::mutex> lock(g_cbMutex);
    g_cbSize.erase(res.handle);
    g_viewData.erase(res.handle);
    g_mapped.erase(res.handle);
}

static void OnMapBuffer(device* dev, resource res, uint64_t offset, uint64_t, map_access access, void** data)
{
    if (access == map_access::read_only || !data || !*data || offset != 0)
        return;
    std::lock_guard<std::mutex> lock(g_cbMutex);
    if (IsViewSized(dev, res))
        g_mapped[res.handle] = *data;
}

// The parts of the View buffer ReplaceTaa reads (offset, size), see NOTES.md.
static const uint32_t kViewParts[][2] = {{416, 16}, {512, 64}, {896, 16}, {944, 8}, {1232, 64}, {1584, 16}, {2128, 16}};

// Whether 4096 bytes are a View buffer: projection w = 1 at 428, a whole TAA sample count at 2132 and
// ViewSizeAndInvSize at 1840 (size * 1/size = 1). Many other 4096-byte buffers pass the first two by chance.
static bool IsViewData(const uint8_t* src)
{
    float w, samples, size[4];
    memcpy(&w, src + 428, 4);
    memcpy(&samples, src + 2132, 4);
    if (w != 1.0f || !(samples >= 1.0f && samples <= 64.0f) || samples != static_cast<float>(static_cast<int>(samples)))
        return false;
    memcpy(size, src + 1840, 16);
    return size[0] >= 320.0f && size[1] >= 180.0f && std::fabs(size[0] * size[2] - 1.0f) < 1e-3f &&
           std::fabs(size[1] * size[3] - 1.0f) < 1e-3f;
}

// Keeps the parts of a View buffer that are read; forgets any other buffer. Caller holds g_cbMutex.
static void StoreViewParts(uint64_t buffer, const uint8_t* src)
{
    if (!IsViewData(src)) {
        if (!g_viewData.empty())
            g_viewData.erase(buffer);
        return;
    }
    auto& v = g_viewData[buffer];
    v.resize(kViewSize);
    for (const auto& part : kViewParts)
        memcpy(v.data() + part[0], src + part[0], part[1]);
}

// A mapped buffer is write-combined memory, very slow to read: UE4 maps a 4096-byte buffer hundreds of times per
// frame in busy scenes, and copying each one whole cost half of the rendering thread. Only a View buffer is copied,
// and only the parts that are read.
static void OnUnmapBuffer(device*, resource res)
{
    std::lock_guard<std::mutex> lock(g_cbMutex);
    auto it = g_mapped.find(res.handle);
    if (it == g_mapped.end())
        return;
    const uint8_t* src = static_cast<const uint8_t*>(it->second);
    g_mapped.erase(it);
    StoreViewParts(res.handle, src);
}

static bool OnUpdateBuffer(device* dev, const void* data, resource res, uint64_t offset, uint64_t size)
{
    if (!data)
        return false;
    std::lock_guard<std::mutex> lock(g_cbMutex);
    if (!IsViewSized(dev, res))
        return false;
    if (size == 0 || size == UINT64_MAX)
        size = kViewSize - offset;
    if (offset + size > kViewSize)
        return false;
    const uint8_t* src = static_cast<const uint8_t*>(data);
    if (offset == 0 && size == kViewSize) {  // whole buffer: as in OnUnmapBuffer
        StoreViewParts(res.handle, src);
        return false;
    }
    auto& v = g_viewData[res.handle];
    v.resize(kViewSize);
    memcpy(v.data() + offset, src, static_cast<size_t>(size));
    return false;
}

// ---------------------------------------------------------------- draw state
// Read from the context at the few draws that can matter (full-screen triangles / quads) instead of tracking every
// bind through ReShade events: the game binds shaders and resources thousands of times per frame, and each event
// costs the rendering thread. In D3D11, ReShade's pipeline and view handles are the native object pointers.

static uint64_t g_rtv0 = 0, g_dsv = 0;
static uint64_t g_psSrv[8] = {};

static void RefreshDrawState(command_list* cmd)
{
    auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(cmd->get_native());
    ID3D11PixelShader* ps = nullptr;
    ctx->PSGetShader(&ps, nullptr, nullptr);
    g_ps = reinterpret_cast<uint64_t>(ps);
    if (ps)
        ps->Release();
    ID3D11ShaderResourceView* srvs[8] = {};
    ctx->PSGetShaderResources(0, 8, srvs);
    for (int i = 0; i < 8; ++i) {
        g_psSrv[i] = reinterpret_cast<uint64_t>(srvs[i]);
        if (srvs[i])
            srvs[i]->Release();
    }
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11DepthStencilView* dsv = nullptr;
    ctx->OMGetRenderTargets(1, &rtv, &dsv);
    g_rtv0 = reinterpret_cast<uint64_t>(rtv);
    g_dsv = reinterpret_cast<uint64_t>(dsv);
    if (rtv)
        rtv->Release();
    if (dsv)
        dsv->Release();
}

// ---------------------------------------------------------------- motion vector pass

struct MotionParams {
    float clipToTranslatedWorld[16];
    float prevTranslatedWorldToClip[16];
    float translationDelta[4];
    float renderSize[2];
    float jitterNdc[2];
    float prevJitterNdc[2];
    float debugMode;
    float pad;
};

struct Pass {
    bool failed = false;
    pipeline_layout layout = {0}, sanitizeLayout = {0}, rcasLayout = {0};
    pipeline pipe = {0}, sanitize = {0}, rcas = {0};
    uint32_t width = 0, height = 0;
    resource motion = {0}, depth = {0}, debug = {0}, color = {0}, final = {0};
    resource_view motionUav = {0}, motionSrv = {0}, depthUav = {0}, depthSrv = {0}, debugUav = {0}, colorUav = {0},
                  finalUav = {0};
    // RCAS output at screen size (EnsureSharpTarget); at native AA in the back buffer's format, with a copy of
    // the back buffer as input
    uint32_t sharpWidth = 0, sharpHeight = 0;
    format sharpFormat = format::unknown;
    resource sharp = {0}, copy = {0};
    resource_view sharpUav = {0}, sharpSrv = {0}, copySrv = {0};
};
static Pass g_pass;
// RCAS before the game's final pass (live, from the overlay): on/off and strength 0..2 (1 = full RCAS).
static bool g_sharpenEnabled = true;
static float g_sharpness = 1.0f;

static float SharpnessNow()
{
    return g_sharpenEnabled ? g_sharpness : 0.0f;
}

static void DestroySharpTarget(device* dev)
{
    if (g_pass.sharpUav.handle) dev->destroy_resource_view(g_pass.sharpUav);
    if (g_pass.sharpSrv.handle) dev->destroy_resource_view(g_pass.sharpSrv);
    if (g_pass.copySrv.handle) dev->destroy_resource_view(g_pass.copySrv);
    if (g_pass.sharp.handle) dev->destroy_resource(g_pass.sharp);
    if (g_pass.copy.handle) dev->destroy_resource(g_pass.copy);
    g_pass.sharpUav = g_pass.sharpSrv = g_pass.copySrv = {0};
    g_pass.sharp = g_pass.copy = {0};
    g_pass.sharpFormat = format::unknown;
    g_pass.sharpWidth = g_pass.sharpHeight = 0;
}

static bool EnsureSharpTarget(device* dev, uint32_t w, uint32_t h, format fmt = format::r16g16b16a16_float,
                              bool withCopy = false)
{
    if (g_pass.sharp.handle && g_pass.sharpWidth == w && g_pass.sharpHeight == h && g_pass.sharpFormat == fmt &&
        (!withCopy || g_pass.copy.handle))
        return true;
    DestroySharpTarget(dev);
    if (!dev->create_resource(resource_desc(w, h, 1, 1, fmt, 1, memory_heap::default_,
                                            resource_usage::unordered_access | resource_usage::shader_resource |
                                                resource_usage::copy_source),
                              nullptr, resource_usage::unordered_access, &g_pass.sharp) ||
        !dev->create_resource_view(g_pass.sharp, resource_usage::unordered_access, resource_view_desc(fmt), &g_pass.sharpUav) ||
        !dev->create_resource_view(g_pass.sharp, resource_usage::shader_resource, resource_view_desc(fmt), &g_pass.sharpSrv) ||
        (withCopy &&
         (!dev->create_resource(resource_desc(w, h, 1, 1, fmt, 1, memory_heap::default_,
                                              resource_usage::shader_resource | resource_usage::copy_dest),
                                nullptr, resource_usage::shader_resource, &g_pass.copy) ||
          !dev->create_resource_view(g_pass.copy, resource_usage::shader_resource, resource_view_desc(fmt), &g_pass.copySrv)))) {
        LOGW("sharpening: failed to create %ux%u target (format %u)", w, h, (unsigned)fmt);
        DestroySharpTarget(dev);
        return false;
    }
    g_pass.sharpWidth = w;
    g_pass.sharpHeight = h;
    g_pass.sharpFormat = fmt;
    return true;
}

static void DestroyPassTextures(device* dev)
{
    for (resource_view v : {g_pass.motionUav, g_pass.motionSrv, g_pass.depthUav, g_pass.depthSrv, g_pass.debugUav,
                            g_pass.colorUav, g_pass.finalUav})
        if (v.handle)
            dev->destroy_resource_view(v);
    for (resource r : {g_pass.motion, g_pass.depth, g_pass.debug, g_pass.color, g_pass.final})
        if (r.handle)
            dev->destroy_resource(r);
    g_pass.motionUav = g_pass.motionSrv = g_pass.depthUav = g_pass.depthSrv = g_pass.debugUav = g_pass.colorUav =
        g_pass.finalUav = {0};
    g_pass.motion = g_pass.depth = g_pass.debug = g_pass.color = g_pass.final = {0};
    g_pass.width = g_pass.height = 0;
    DestroySharpTarget(dev);
}

static bool EnsurePass(device* dev, uint32_t w, uint32_t h)
{
    if (g_pass.failed)
        return false;
    if (!g_pass.pipe.handle) {
        descriptor_range srvs = {};
        srvs.count = 3;
        srvs.visibility = shader_stage::compute;
        srvs.type = descriptor_type::shader_resource_view;
        descriptor_range uavs = {};
        uavs.count = 4;
        uavs.visibility = shader_stage::compute;
        uavs.type = descriptor_type::unordered_access_view;
        constant_range consts = {};
        consts.count = sizeof(MotionParams) / 4;
        consts.visibility = shader_stage::compute;
        const pipeline_layout_param params[3] = {pipeline_layout_param(srvs), pipeline_layout_param(uavs),
                                                 pipeline_layout_param(consts)};
        shader_desc cs = {};
        cs.code = g_MotionCS;
        cs.code_size = sizeof(g_MotionCS);
        pipeline_subobject sub = {pipeline_subobject_type::compute_shader, 1, &cs};
        // Sanitize pass: one SRV in, one UAV out.
        descriptor_range sIn = srvs, sOut = uavs;
        sIn.count = 1;
        sOut.count = 1;
        const pipeline_layout_param sParams[2] = {pipeline_layout_param(sIn), pipeline_layout_param(sOut)};
        shader_desc scs = {};
        scs.code = g_SanitizeCS;
        scs.code_size = sizeof(g_SanitizeCS);
        pipeline_subobject ssub = {pipeline_subobject_type::compute_shader, 1, &scs};
        // RCAS pass: one SRV in, one UAV out, sharpness as constant.
        constant_range rConsts = consts;
        rConsts.count = 4;
        const pipeline_layout_param rParams[3] = {pipeline_layout_param(sIn), pipeline_layout_param(sOut),
                                                  pipeline_layout_param(rConsts)};
        shader_desc rcs = {};
        rcs.code = g_RcasCS;
        rcs.code_size = sizeof(g_RcasCS);
        pipeline_subobject rsub = {pipeline_subobject_type::compute_shader, 1, &rcs};
        if (!dev->create_pipeline_layout(3, params, &g_pass.layout) || !dev->create_pipeline(g_pass.layout, 1, &sub, &g_pass.pipe) ||
            !dev->create_pipeline_layout(2, sParams, &g_pass.sanitizeLayout) ||
            !dev->create_pipeline(g_pass.sanitizeLayout, 1, &ssub, &g_pass.sanitize) ||
            !dev->create_pipeline_layout(3, rParams, &g_pass.rcasLayout) ||
            !dev->create_pipeline(g_pass.rcasLayout, 1, &rsub, &g_pass.rcas)) {
            LOGW("motion vectors: failed to create compute pipelines");
            g_pass.failed = true;
            return false;
        }
    }
    if (g_pass.width == w && g_pass.height == h)
        return true;
    DestroyPassTextures(dev);
    const resource_usage rw = resource_usage::unordered_access | resource_usage::shader_resource;
    if (!dev->create_resource(resource_desc(w, h, 1, 1, format::r16g16_float, 1, memory_heap::default_, rw), nullptr,
                              resource_usage::unordered_access, &g_pass.motion) ||
        !dev->create_resource(resource_desc(w, h, 1, 1, format::r32_float, 1, memory_heap::default_, rw), nullptr,
                              resource_usage::unordered_access, &g_pass.depth) ||
        !dev->create_resource(resource_desc(w, h, 1, 1, format::r16g16b16a16_float, 1, memory_heap::default_,
                                            rw | resource_usage::copy_source),
                              nullptr, resource_usage::unordered_access, &g_pass.debug) ||
        !dev->create_resource_view(g_pass.motion, resource_usage::unordered_access, resource_view_desc(format::r16g16_float), &g_pass.motionUav) ||
        !dev->create_resource_view(g_pass.motion, resource_usage::shader_resource, resource_view_desc(format::r16g16_float), &g_pass.motionSrv) ||
        !dev->create_resource_view(g_pass.depth, resource_usage::unordered_access, resource_view_desc(format::r32_float), &g_pass.depthUav) ||
        !dev->create_resource_view(g_pass.depth, resource_usage::shader_resource, resource_view_desc(format::r32_float), &g_pass.depthSrv) ||
        !dev->create_resource_view(g_pass.debug, resource_usage::unordered_access, resource_view_desc(format::r16g16b16a16_float), &g_pass.debugUav) ||
        !dev->create_resource(resource_desc(w, h, 1, 1, format::r16g16b16a16_float, 1, memory_heap::default_,
                                            rw | resource_usage::copy_source),
                              nullptr, resource_usage::unordered_access, &g_pass.color) ||
        !dev->create_resource_view(g_pass.color, resource_usage::unordered_access, resource_view_desc(format::r16g16b16a16_float), &g_pass.colorUav) ||
        !dev->create_resource(resource_desc(w, h, 1, 1, format::r16g16b16a16_float, 1, memory_heap::default_,
                                            rw | resource_usage::copy_source),
                              nullptr, resource_usage::unordered_access, &g_pass.final) ||
        !dev->create_resource_view(g_pass.final, resource_usage::unordered_access, resource_view_desc(format::r16g16b16a16_float), &g_pass.finalUav)) {
        LOGW("motion vectors: failed to create %ux%u textures", w, h);
        DestroyPassTextures(dev);
        g_pass.failed = true;
        return false;
    }
    g_pass.width = w;
    g_pass.height = h;
    LOGI("motion vectors: textures %ux%u ready", w, h);
    return true;
}

// ---------------------------------------------------------------- upscaler (NGX interface, answered by OptiScaler)
// NVIDIA's NGX loader (nvsdk_ngx_s.lib) refuses to start on a non-NVIDIA adapter unless OptiScaler spoofs
// the GPU, and spoofing breaks this game's renderer (UE4 then takes NVIDIA-only paths). OptiScaler exports
// the NGX entry points itself, so they are called on its module directly.

// The exported NVSDK_NGX_D3D11_Init is the 5-parameter form (with NVSDK_NGX_FeatureCommonInfo).
using PFN_Init = NVSDK_NGX_Result(NVSDK_CONV*)(unsigned long long, const wchar_t*, ID3D11Device*,
                                                const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version);
using PFN_GetCaps = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**);
using PFN_DestroyParams = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*);
using PFN_Create = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D11DeviceContext*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
using PFN_Evaluate = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D11DeviceContext*, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, void*);
using PFN_Release = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
using PFN_Shutdown = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D11Device*);

static struct {
    PFN_Init init = nullptr;
    PFN_GetCaps getCaps = nullptr;
    PFN_DestroyParams destroyParams = nullptr;
    PFN_Create create = nullptr;
    PFN_Evaluate evaluate = nullptr;
    PFN_Release release = nullptr;
    PFN_Shutdown shutdown = nullptr;
} g_ngx;

static void WriteUpscalerChoiceToOptiScaler();

// OptiScaler can be installed under several proxy names; take the one in the game folder that exports NGX.
static bool LoadNgx()
{
    if (g_ngx.init)
        return true;
    wchar_t dir[MAX_PATH];
    GetModuleFileNameW(nullptr, dir, MAX_PATH);
    if (wchar_t* slash = wcsrchr(dir, L'\\'))
        slash[1] = L'\0';
    // Preferred: OptiScaler as nvngx.dll, loaded only now. The game then starts without OptiScaler's hooks on
    // device and swap chain creation, which break this game's rendering when installed as a proxy DLL.
    for (const wchar_t* name : {L"nvngx.dll", L"winmm.dll", L"version.dll", L"dbghelp.dll", L"wininet.dll", L"winhttp.dll", L"OptiScaler.asi"}) {
        const std::wstring path = std::wstring(dir) + name;
        HMODULE m = GetModuleHandleW(path.c_str());
        if (!m && wcscmp(name, L"nvngx.dll") == 0 && GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES) {
            WriteUpscalerChoiceToOptiScaler();  // OptiScaler reads its ini when it loads
            m = LoadLibraryW(path.c_str());
        }
        if (!m || !GetProcAddress(m, "NVSDK_NGX_D3D11_Init"))
            continue;
        g_ngx.init = reinterpret_cast<PFN_Init>(GetProcAddress(m, "NVSDK_NGX_D3D11_Init"));
        g_ngx.getCaps = reinterpret_cast<PFN_GetCaps>(GetProcAddress(m, "NVSDK_NGX_D3D11_GetCapabilityParameters"));
        g_ngx.destroyParams = reinterpret_cast<PFN_DestroyParams>(GetProcAddress(m, "NVSDK_NGX_D3D11_DestroyParameters"));
        g_ngx.create = reinterpret_cast<PFN_Create>(GetProcAddress(m, "NVSDK_NGX_D3D11_CreateFeature"));
        g_ngx.evaluate = reinterpret_cast<PFN_Evaluate>(GetProcAddress(m, "NVSDK_NGX_D3D11_EvaluateFeature"));
        g_ngx.release = reinterpret_cast<PFN_Release>(GetProcAddress(m, "NVSDK_NGX_D3D11_ReleaseFeature"));
        g_ngx.shutdown = reinterpret_cast<PFN_Shutdown>(GetProcAddress(m, "NVSDK_NGX_D3D11_Shutdown1"));
        if (g_ngx.getCaps && g_ngx.destroyParams && g_ngx.create && g_ngx.evaluate && g_ngx.release && g_ngx.shutdown) {
            char n[64];
            WideCharToMultiByte(CP_UTF8, 0, name, -1, n, sizeof(n), nullptr, nullptr);
            LOGI("upscaler: using OptiScaler's NGX interface (%s)", n);
            return true;
        }
        g_ngx = {};
    }
    return false;
}

// Calls into OptiScaler behind an exception guard: a fault there must not take the game down. Returns
// false (and logs where) on an access violation or other structured exception.
static DWORD g_faultCode = 0;
static void* g_faultAddress = nullptr;
static int Guard(DWORD code, EXCEPTION_POINTERS* ep)
{
    g_faultCode = code;
    g_faultAddress = ep->ExceptionRecord->ExceptionAddress;
    return EXCEPTION_EXECUTE_HANDLER;
}

static NVSDK_NGX_Result GuardedInit(ID3D11Device* d3d, const wchar_t* dir)
{
    __try {
        static const NVSDK_NGX_FeatureCommonInfo info = {};
        return g_ngx.init(0x5354595832ull, dir, d3d, &info, NVSDK_NGX_Version_API);
    } __except (Guard(GetExceptionCode(), GetExceptionInformation())) {
        return NVSDK_NGX_Result_Fail;
    }
}

static NVSDK_NGX_Result GuardedGetCaps(NVSDK_NGX_Parameter** out)
{
    __try {
        return g_ngx.getCaps(out);
    } __except (Guard(GetExceptionCode(), GetExceptionInformation())) {
        return NVSDK_NGX_Result_Fail;
    }
}

static void LogFault(const char* what)
{
    if (!g_faultCode)
        return;
    HMODULE m = nullptr;
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(g_faultAddress), &m))
        GetModuleFileNameA(m, name, MAX_PATH);
    const char* base = strrchr(name, '\\');
    LOGW("upscaler: %s crashed (exception 0x%08lx at %s+0x%llx)", what, g_faultCode, base ? base + 1 : name,
         static_cast<unsigned long long>(static_cast<uint8_t*>(g_faultAddress) - reinterpret_cast<uint8_t*>(m)));
    g_faultCode = 0;
}

struct Upscaler {
    bool initTried = false, available = false;
    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    uint32_t width = 0, height = 0;              // output size
    uint32_t renderWidth = 0, renderHeight = 0;  // input size
    bool hdr = true;
    resource output = {0};
    resource_view outputSrv = {0};
    bool resetNext = true;
    unsigned evaluations = 0, failures = 0;
};
static Upscaler g_up;

// Retried every few seconds: the first TAA frames can come before OptiScaler is ready.
static bool InitUpscaler(device* dev)
{
    static ULONGLONG lastTry = 0;
    if (g_up.available)
        return true;
    if (g_up.initTried && GetTickCount64() - lastTry < 5000)
        return false;
    g_up.initTried = true;
    lastTry = GetTickCount64();
    {
        IDXGIDevice* dxgi = nullptr;
        IDXGIAdapter* adapter = nullptr;
        DXGI_ADAPTER_DESC ad = {};
        if (SUCCEEDED(reinterpret_cast<ID3D11Device*>(dev->get_native())->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgi))) &&
            SUCCEEDED(dxgi->GetAdapter(&adapter)))
            adapter->GetDesc(&ad);
        if (adapter) adapter->Release();
        if (dxgi) dxgi->Release();
        LOGI("upscaler: device adapter vendor 0x%04x", ad.VendorId);
    }
    if (!LoadNgx()) {
        LOGW("upscaler: OptiScaler not found - install it as winmm.dll next to the game exe");
        return false;
    }
    wchar_t dir[MAX_PATH];
    GetModuleFileNameW(nullptr, dir, MAX_PATH);
    if (wchar_t* slash = wcsrchr(dir, L'\\'))
        slash[0] = L'\0';
    auto* d3d = reinterpret_cast<ID3D11Device*>(dev->get_native());
    NVSDK_NGX_Result r = GuardedInit(d3d, dir);
    LogFault("NVSDK_NGX_D3D11_Init");
    if (NVSDK_NGX_FAILED(r)) {
        LOGW("upscaler: NVSDK_NGX_D3D11_Init failed (0x%08x)", (unsigned)r);
        return false;
    }
    r = GuardedGetCaps(&g_up.params);
    LogFault("NVSDK_NGX_D3D11_GetCapabilityParameters");
    int available = 0;
    if (NVSDK_NGX_SUCCEED(r) && g_up.params)
        g_up.params->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &available);
    if (!available) {
        LOGW("upscaler: super sampling unavailable (0x%08x)", (unsigned)r);
        return false;
    }
    g_up.available = true;
    LOGI("upscaler: NGX initialised, super sampling available");
    return true;
}

static void ShutdownUpscaler(device* dev)
{
    if (g_up.feature)
        g_ngx.release(g_up.feature);
    if (g_up.params)
        g_ngx.destroyParams(g_up.params);
    if (g_up.available)
        g_ngx.shutdown(reinterpret_cast<ID3D11Device*>(dev->get_native()));
    if (g_up.outputSrv.handle)
        dev->destroy_resource_view(g_up.outputSrv);
    if (g_up.output.handle)
        dev->destroy_resource(g_up.output);
    g_up = {};
}

// rw x rh -> ow x oh. hdr: native AA at the TAA pass (scene color, linear HDR); otherwise upscaling of the
// tone-mapped image at the game's final upscale pass.
static bool EnsureFeature(command_list* cmd, uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh, bool hdr)
{
    if (g_up.feature && g_up.renderWidth == rw && g_up.renderHeight == rh && g_up.width == ow && g_up.height == oh &&
        g_up.hdr == hdr)
        return true;
    const uint32_t w = ow, h = oh;
    device* dev = cmd->get_device();
    if (g_up.feature) {
        g_ngx.release(g_up.feature);
        g_up.feature = nullptr;
    }
    if (g_up.outputSrv.handle)
        dev->destroy_resource_view(g_up.outputSrv);
    if (g_up.output.handle)
        dev->destroy_resource(g_up.output);
    g_up.output = {0};
    g_up.outputSrv = {0};
    resource_desc od(w, h, 1, 1, format::r16g16b16a16_float, 1, memory_heap::default_,
                     resource_usage::unordered_access | resource_usage::shader_resource | resource_usage::copy_source);
    if (!dev->create_resource(od, nullptr, resource_usage::unordered_access, &g_up.output) ||
        !dev->create_resource_view(g_up.output, resource_usage::shader_resource, resource_view_desc(format::r16g16b16a16_float), &g_up.outputSrv)) {
        LOGW("upscaler: failed to create %ux%u output", w, h);
        return false;
    }
    NVSDK_NGX_Parameter* p = g_up.params;
    p->Set(NVSDK_NGX_Parameter_Width, rw);
    p->Set(NVSDK_NGX_Parameter_Height, rh);
    p->Set(NVSDK_NGX_Parameter_OutWidth, ow);
    p->Set(NVSDK_NGX_Parameter_OutHeight, oh);
    p->Set(NVSDK_NGX_Parameter_PerfQualityValue,
           static_cast<int>(rw >= ow ? NVSDK_NGX_PerfQuality_Value_DLAA : NVSDK_NGX_PerfQuality_Value_MaxQuality));
    const int flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_DepthInverted |  // UE4: reversed Z
                      (hdr ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure : 0);
    p->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);
    p->Set(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
    auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(cmd->get_native());
    NVSDK_NGX_Result r = g_ngx.create(ctx, NVSDK_NGX_Feature_SuperSampling, p, &g_up.feature);
    if (NVSDK_NGX_FAILED(r) || !g_up.feature) {
        LOGW("upscaler: create feature %ux%u -> %ux%u failed (0x%08x)", rw, rh, ow, oh, (unsigned)r);
        g_up.feature = nullptr;
        return false;
    }
    g_up.renderWidth = rw;
    g_up.renderHeight = rh;
    g_up.width = ow;
    g_up.height = oh;
    g_up.hdr = hdr;
    g_up.resetNext = true;
    LOGI("upscaler: feature created %ux%u -> %ux%u (%s)", rw, rh, ow, oh, hdr ? "native AA, HDR" : "upscaling, tone-mapped");
    return true;
}

// ---------------------------------------------------------------- TAA replacement

static uint32_t g_backBufferWidth = 0, g_backBufferHeight = 0;

// ---------------------------------------------------------------- latency
// [Latency] AntiLag2: AMD Anti-Lag 2 (Radeon, driver 24.6.1+, DX11 through amdxx64.dll). Its Update() belongs
// right before the game polls input; the closest point an injected mod has is the present of the previous frame.
// [Latency] MaxFrameLatency: fallback when Anti-Lag 2 is unavailable (NVIDIA, Intel): frames the CPU may queue
// ahead of the GPU (DXGI default 3); 0 leaves the game's setting alone.

static bool g_antiLag2Wanted = true;
static int g_maxFrameLatency = 1;
static AMD::AntiLag2DX11::Context g_antiLag2 = {};
static enum { kLatencyPending, kLatencyAntiLag2, kLatencyFallback } g_latencyMode = kLatencyPending;
static ID3D11Device* g_latencyDevice = nullptr;  // device of the swap chain; not owned

// Once per frame, at present. The driver module is only loaded once the device renders, so Anti-Lag 2 is
// initialised on the first frame rather than at device creation.
static void LatencyFrameEnd()
{
    if (!g_latencyDevice)
        return;
    if (g_latencyMode == kLatencyPending) {
        const HRESULT hr = g_antiLag2Wanted ? AMD::AntiLag2DX11::Initialize(&g_antiLag2) : E_ABORT;
        if (hr == S_OK) {
            g_latencyMode = kLatencyAntiLag2;
            LOGI("latency: AMD Anti-Lag 2 active");
        } else {
            g_latencyMode = kLatencyFallback;
            if (g_antiLag2Wanted)
                LOGI("latency: AMD Anti-Lag 2 not available (0x%08x), using the frame queue limit", static_cast<unsigned>(hr));
            IDXGIDevice1* dxgi = nullptr;
            if (g_maxFrameLatency > 0 &&
                SUCCEEDED(g_latencyDevice->QueryInterface(__uuidof(IDXGIDevice1), reinterpret_cast<void**>(&dxgi)))) {
                dxgi->SetMaximumFrameLatency(static_cast<UINT>(g_maxFrameLatency));
                dxgi->Release();
                LOGI("latency: maximum frame latency set to %d", g_maxFrameLatency);
            }
        }
    }
    if (g_latencyMode == kLatencyAntiLag2)
        AMD::AntiLag2DX11::Update(&g_antiLag2, g_antiLag2Wanted, 0);  // switchable in the overlay
}

static void ShutdownLatency(device* dev)
{
    if (!g_latencyDevice || reinterpret_cast<ID3D11Device*>(dev->get_native()) != g_latencyDevice)
        return;
    if (g_latencyMode == kLatencyAntiLag2)
        AMD::AntiLag2DX11::DeInitialize(&g_antiLag2);
    g_latencyDevice = nullptr;
    g_latencyMode = kLatencyPending;
}

// Back buffer resources of the presenting swap chain: the game's final pass is the draw into one of them (at
// native AA every scene target has the screen size, so the size alone does not identify it).
static std::vector<uint64_t> g_backBuffers;

static void OnInitSwapchain(swapchain* sc, bool)
{
    resource_desc d = sc->get_device()->get_resource_desc(sc->get_current_back_buffer());
    g_backBufferWidth = d.texture.width;
    g_backBufferHeight = d.texture.height;
    g_backBuffers.clear();
    for (uint32_t i = 0; i < sc->get_back_buffer_count(); ++i)
        g_backBuffers.push_back(sc->get_back_buffer(i).handle);
    LOGI("swap chain %ux%u, %zu back buffers", g_backBufferWidth, g_backBufferHeight, g_backBuffers.size());
    // Anti-Lag 2 / frame latency follow the device that presents (helper devices come and go).
    if (sc->get_device()->get_api() == device_api::d3d11)
        g_latencyDevice = reinterpret_cast<ID3D11Device*>(sc->get_device()->get_native());
}

static uint32_t g_frameId = 1;
static uint32_t g_replacedFrame = 0;  // frame whose TAA draws we replaced (both of them are skipped)
static uint32_t g_checkedFrame = 0;
static unsigned g_replacements = 0, g_viewMisses = 0;

static float ReadF(const std::vector<uint8_t>& v, uint32_t offset)
{
    float f;
    memcpy(&f, v.data() + offset, 4);
    return f;
}

// UE 4.13's TAA pixel shaders in this game (FNV-1a of the bytecode): main pass and responsive-AA pass.
static const uint64_t kTaaShaders[] = {0x8f74141e5b1d3aa6ull, 0x741f2eb17664c147ull};

// The TAA pass: one of the TAA shaders drawing a full-screen triangle into an RGBA16F target, reading depth
// (t0, R24G8) and velocity (t3, R16G16).
static bool IsTaaDraw(device* dev, uint32_t indexCount)
{
    if (indexCount != 3 || !g_rtv0 || !g_dsv || !g_psSrv[0] || !g_psSrv[1] || !g_psSrv[3])
        return false;
    const uint64_t ps = PsHash(g_ps);
    if (ps != kTaaShaders[0] && ps != kTaaShaders[1])
        return false;
    const ViewInfo rt = GetViewInfo(dev, g_rtv0);
    if (rt.fmt != format::r16g16b16a16_float)
        return false;
    return GetViewInfo(dev, g_psSrv[3]).fmt == format::r16g16_unorm && GetViewInfo(dev, g_psSrv[0]).fmt == format::r24_g8_typeless;
}

// Work left for the game's final pass (tone-mapped image -> back buffer, UI follows):
// - upscaling (render resolution below the screen): the TAA pass only builds motion vectors and passes the raw
//   image on; FSR runs on the tone-mapped image at the final pass;
// - sharpening: RCAS on the tone-mapped image (also at native AA).
static struct {
    uint32_t frame = 0;  // frame the TAA pass ran in
    uint32_t width = 0, height = 0;  // size of the final pass's input (render size)
    bool upscale = false;
    float jitterNdc[2] = {};
} g_pending;

// UE4's D3D11 RHI caches the bound pipeline state and skips "redundant" sets, so anything the add-on leaves
// bound (compute shader, constant buffer, views) would be used by the game's next pass - e.g. eye adaptation
// running with our constants. All add-on work therefore runs in its own device context state, and the game's
// exact state is swapped back afterwards.
static ID3DDeviceContextState* g_ownState = nullptr;

static void ReleaseOwnState()
{
    if (g_ownState)
        g_ownState->Release();
    g_ownState = nullptr;
}

class ScopedOwnState {
public:
    explicit ScopedOwnState(ID3D11DeviceContext* ctx)
    {
        if (!g_ownState) {
            ID3D11Device* dev = nullptr;
            ID3D11Device1* dev1 = nullptr;
            ctx->GetDevice(&dev);
            const D3D_FEATURE_LEVEL level = dev->GetFeatureLevel();
            if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D11Device1), reinterpret_cast<void**>(&dev1)))) {
                dev1->CreateDeviceContextState(0, &level, 1, D3D11_SDK_VERSION, __uuidof(ID3D11Device), nullptr, &g_ownState);
                dev1->Release();
            }
            dev->Release();
        }
        if (g_ownState && SUCCEEDED(ctx->QueryInterface(__uuidof(ID3D11DeviceContext1), reinterpret_cast<void**>(&ctx1_))))
            ctx1_->SwapDeviceContextState(g_ownState, &saved_);
    }
    ~ScopedOwnState()
    {
        if (ctx1_ && saved_) {
            ctx1_->SwapDeviceContextState(saved_, nullptr);
            saved_->Release();
        }
        if (ctx1_)
            ctx1_->Release();
    }
    bool ok() const { return saved_ != nullptr; }

private:
    ID3D11DeviceContext1* ctx1_ = nullptr;
    ID3DDeviceContextState* saved_ = nullptr;
};

// Copies `in` to `out` (both w x h), replacing NaN / Inf. Caller is inside ScopedOwnState.
static void Sanitize(command_list* cmd, resource_view in, resource_view out, uint32_t w, uint32_t h)
{
    const resource_view none = {0};
    cmd->bind_pipeline(pipeline_stage::compute_shader, g_pass.sanitize);
    cmd->push_descriptors(shader_stage::compute, g_pass.sanitizeLayout, 0, descriptor_table_update{{}, 0, 0, 1, descriptor_type::shader_resource_view, &in});
    cmd->push_descriptors(shader_stage::compute, g_pass.sanitizeLayout, 1, descriptor_table_update{{}, 0, 0, 1, descriptor_type::unordered_access_view, &out});
    cmd->dispatch((w + 7) / 8, (h + 7) / 8, 1);
    cmd->push_descriptors(shader_stage::compute, g_pass.sanitizeLayout, 0, descriptor_table_update{{}, 0, 0, 1, descriptor_type::shader_resource_view, &none});
    cmd->push_descriptors(shader_stage::compute, g_pass.sanitizeLayout, 1, descriptor_table_update{{}, 0, 0, 1, descriptor_type::unordered_access_view, &none});
}

// One upscaler evaluation: `color` (render size w x h) with our depth and motion vectors -> g_up.output.
static bool Evaluate(ID3D11DeviceContext* ctx, resource color, uint32_t w, uint32_t h, const float jitterNdc[2])
{
    NVSDK_NGX_Parameter* ep = g_up.params;
    ep->Set(NVSDK_NGX_Parameter_Color, reinterpret_cast<ID3D11Resource*>(color.handle));
    ep->Set(NVSDK_NGX_Parameter_Output, reinterpret_cast<ID3D11Resource*>(g_up.output.handle));
    ep->Set(NVSDK_NGX_Parameter_Depth, reinterpret_cast<ID3D11Resource*>(g_pass.depth.handle));
    ep->Set(NVSDK_NGX_Parameter_MotionVectors, reinterpret_cast<ID3D11Resource*>(g_pass.motion.handle));
    // Jitter in render pixels: the projection got clip.xy += (2*jx/W, -2*jy/H) * clip.w.
    ep->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, jitterNdc[0] * 0.5f * static_cast<float>(w));
    ep->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, -jitterNdc[1] * 0.5f * static_cast<float>(h));
    ep->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f);
    ep->Set(NVSDK_NGX_Parameter_Reset, g_up.resetNext ? 1 : 0);
    ep->Set(NVSDK_NGX_Parameter_MV_Scale_X, static_cast<float>(w));  // our vectors are in UV units
    ep->Set(NVSDK_NGX_Parameter_MV_Scale_Y, static_cast<float>(h));
    ep->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, w);
    ep->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, h);
    ep->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
    ep->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    NVSDK_NGX_Result r = g_ngx.evaluate(ctx, g_up.feature, ep, nullptr);
    if (NVSDK_NGX_FAILED(r)) {
        if (g_up.failures++ < 5)
            LOGW("upscaler: evaluate failed (0x%08x)", (unsigned)r);
        return false;
    }
    g_up.resetNext = false;
    ++g_up.evaluations;
    return true;
}

static bool ReplaceTaa(command_list* cmd)
{
    device* dev = cmd->get_device();
    auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(cmd->get_native());
    const ViewInfo out = GetViewInfo(dev, g_rtv0);
    const ViewInfo color = GetViewInfo(dev, g_psSrv[1]);
    const uint32_t w = out.width, h = out.height;
    if (color.width != w || color.height != h)
        return false;

    // Camera and jitter from the View uniform buffer bound at b1.
    ID3D11Buffer* viewCb = nullptr;
    ctx->PSGetConstantBuffers(1, 1, &viewCb);
    if (viewCb)
        viewCb->Release();
    std::vector<uint8_t> view;
    {
        std::lock_guard<std::mutex> lock(g_cbMutex);
        auto it = g_viewData.find(reinterpret_cast<uintptr_t>(viewCb));
        if (it != g_viewData.end())
            view = it->second;
    }
    // Sanity: ViewToClip row 2 ends in 1 (perspective), TemporalAAParams.y = sample count.
    if (view.size() < kViewSize || ReadF(view, 428) != 1.0f || ReadF(view, 2132) < 1.0f) {
        if (g_viewMisses++ < 5)
            LOGW("TAA replacement: View constants not found (buffer %p, %zu bytes)", static_cast<void*>(viewCb), view.size());
        return false;
    }

    // From here on the game's pipeline state is parked; it comes back when `own` goes out of scope. This also
    // unbinds the game's depth-stencil, so its depth buffer can be read as a texture.
    ScopedOwnState own(ctx);
    if (!own.ok()) {
        if (g_viewMisses++ < 5)
            LOGW("TAA replacement: could not switch device context state");
        return false;
    }
    // Render resolution below the screen (quality presets): FSR upscales later, at the final upscale pass.
    const bool upscaling = w < g_backBufferWidth && h < g_backBufferHeight;
    if (!EnsurePass(dev, w, h))
        return false;
    if (!g_debugView && (!InitUpscaler(dev) || (!upscaling && !EnsureFeature(cmd, w, h, w, h, true))))
        return false;  // the debug views work without the upscaler

    MotionParams p = {};
    memcpy(p.clipToTranslatedWorld, view.data() + 512, 64);
    memcpy(p.prevTranslatedWorldToClip, view.data() + 1232, 64);
    for (int i = 0; i < 3; ++i)
        p.translationDelta[i] = ReadF(view, 1584 + 4 * i) - ReadF(view, 896 + 4 * i);
    p.renderSize[0] = static_cast<float>(w);
    p.renderSize[1] = static_cast<float>(h);
    p.jitterNdc[0] = ReadF(view, 416);
    p.jitterNdc[1] = ReadF(view, 420);
    p.prevJitterNdc[0] = ReadF(view, 944);
    p.prevJitterNdc[1] = ReadF(view, 948);
    p.debugMode = static_cast<float>(g_debugView);

    const resource_view srvs[3] = {{g_psSrv[0]}, {g_psSrv[3]}, {g_psSrv[1]}};
    const resource_view uavs[4] = {g_pass.motionUav, g_pass.depthUav, g_pass.debugUav, g_pass.colorUav};
    const resource_view none[4] = {{0}, {0}, {0}, {0}};
    cmd->bind_pipeline(pipeline_stage::compute_shader, g_pass.pipe);
    cmd->push_descriptors(shader_stage::compute, g_pass.layout, 0, descriptor_table_update{{}, 0, 0, 3, descriptor_type::shader_resource_view, srvs});
    cmd->push_descriptors(shader_stage::compute, g_pass.layout, 1, descriptor_table_update{{}, 0, 0, 4, descriptor_type::unordered_access_view, uavs});
    cmd->push_constants(shader_stage::compute, g_pass.layout, 2, 0, sizeof(MotionParams) / 4, &p);
    cmd->dispatch((w + 7) / 8, (h + 7) / 8, 1);
    cmd->push_descriptors(shader_stage::compute, g_pass.layout, 0, descriptor_table_update{{}, 0, 0, 3, descriptor_type::shader_resource_view, none});
    cmd->push_descriptors(shader_stage::compute, g_pass.layout, 1, descriptor_table_update{{}, 0, 0, 4, descriptor_type::unordered_access_view, none});

    if (g_debugView == 1 || g_debugView == 3) {  // motion vectors / velocity check
        cmd->copy_resource(g_pass.debug, resource{out.res});
        return true;
    }
    if (g_debugView == 2 || upscaling) {
        // Pass-through (debug view) or upscaling: the raw jittered image goes on through bloom and tone mapping.
        cmd->copy_resource(g_pass.color, resource{out.res});
        if (upscaling && !g_debugView) {
            g_pending.frame = g_frameId;
            g_pending.width = w;
            g_pending.height = h;
            g_pending.upscale = true;
            g_pending.jitterNdc[0] = p.jitterNdc[0];
            g_pending.jitterNdc[1] = p.jitterNdc[1];
        }
        return true;
    }

    if (!Evaluate(ctx, g_pass.color, w, h, p.jitterNdc))
        return false;
    // Back into the game's TAA target, without NaN / Inf (they would reach bloom and eye adaptation).
    Sanitize(cmd, g_up.outputSrv, g_pass.finalUav, w, h);
    cmd->copy_resource(g_pass.final, resource{out.res});
    g_pending.frame = g_frameId;  // the final pass may still sharpen
    g_pending.width = w;
    g_pending.height = h;
    g_pending.upscale = false;
    return true;
}

static bool DispatchRcas(command_list* cmd, resource_view in, uint32_t w, uint32_t h);

// RCAS from `in` (tone-mapped, w x h) into g_pass.sharp. Caller is inside ScopedOwnState.
static bool Sharpen(command_list* cmd, resource_view in, uint32_t w, uint32_t h)
{
    if (!EnsureSharpTarget(cmd->get_device(), w, h))
        return false;
    return DispatchRcas(cmd, in, w, h);
}

static bool DispatchRcas(command_list* cmd, resource_view in, uint32_t w, uint32_t h)
{
    const resource_view none = {0};
    const float consts[4] = {SharpnessNow(), 0.0f, 0.0f, 0.0f};
    cmd->bind_pipeline(pipeline_stage::compute_shader, g_pass.rcas);
    cmd->push_descriptors(shader_stage::compute, g_pass.rcasLayout, 0, descriptor_table_update{{}, 0, 0, 1, descriptor_type::shader_resource_view, &in});
    cmd->push_descriptors(shader_stage::compute, g_pass.rcasLayout, 1, descriptor_table_update{{}, 0, 0, 1, descriptor_type::unordered_access_view, &g_pass.sharpUav});
    cmd->push_constants(shader_stage::compute, g_pass.rcasLayout, 2, 0, 4, consts);
    cmd->dispatch((w + 7) / 8, (h + 7) / 8, 1);
    cmd->push_descriptors(shader_stage::compute, g_pass.rcasLayout, 0, descriptor_table_update{{}, 0, 0, 1, descriptor_type::shader_resource_view, &none});
    cmd->push_descriptors(shader_stage::compute, g_pass.rcasLayout, 1, descriptor_table_update{{}, 0, 0, 1, descriptor_type::unordered_access_view, &none});
    return true;
}

// The game's final pass (tone-mapped image -> back buffer, UI follows): upscale that image with FSR and/or
// sharpen it with RCAS, and let the game's pass read the result instead, 1:1.
static bool ReplaceFinalPass(command_list* cmd, bool indexed, uint32_t count, uint32_t instances, uint32_t first,
                             int32_t vertexOffset, uint32_t firstInstance)
{
    auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(cmd->get_native());
    const uint32_t rw = g_pending.width, rh = g_pending.height;
    resource_view image = {g_psSrv[0]};  // what the game's pass will read
    {
        ScopedOwnState own(ctx);
        if (!own.ok())
            return false;
        if (g_pending.upscale) {
            if (!EnsureFeature(cmd, rw, rh, g_backBufferWidth, g_backBufferHeight, false))
                return false;
            Sanitize(cmd, image, g_pass.colorUav, rw, rh);  // also converts to RGBA16F
            if (!Evaluate(ctx, g_pass.color, rw, rh, g_pending.jitterNdc))
                return false;
            image = g_up.outputSrv;
        }
        const uint32_t w = g_pending.upscale ? g_backBufferWidth : rw, h = g_pending.upscale ? g_backBufferHeight : rh;
        if (SharpnessNow() > 0.0f && Sharpen(cmd, image, w, h))
            image = g_pass.sharpSrv;
        else if (!g_pending.upscale)
            return false;  // nothing to do: let the game draw
    }
    // The game's state is back: draw its pass once more with our image in t0, then restore t0.
    ID3D11ShaderResourceView* gameSrv = nullptr;
    ctx->PSGetShaderResources(0, 1, &gameSrv);
    ID3D11ShaderResourceView* ours = reinterpret_cast<ID3D11ShaderResourceView*>(image.handle);
    ctx->PSSetShaderResources(0, 1, &ours);
    if (indexed)
        ctx->DrawIndexedInstanced(count, instances, first, vertexOffset, firstInstance);
    else
        ctx->DrawInstanced(count, instances, first, firstInstance);
    ctx->PSSetShaderResources(0, 1, &gameSrv);
    if (gameSrv)
        gameSrv->Release();
    return true;
}

// The final pass: full-screen draw into the back buffer whose t0 has the render resolution, in a frame whose
// TAA pass left work for it.
static bool IsFinalUpscaleDraw(command_list* cmd, uint32_t count)
{
    if (g_pending.frame != g_frameId || count > 6)
        return false;
    if (!g_pending.upscale && SharpnessNow() <= 0.0f)
        return false;
    device* dev = cmd->get_device();
    RefreshDrawState(cmd);
    if (!g_rtv0 || !g_psSrv[0])
        return false;
    const ViewInfo rt = GetViewInfo(dev, g_rtv0);
    if (std::find(g_backBuffers.begin(), g_backBuffers.end(), rt.res) == g_backBuffers.end())
        return false;
    if (!g_pending.upscale)  // native AA: the tone mapper itself draws into the back buffer
        return true;
    const ViewInfo src = GetViewInfo(dev, g_psSrv[0]);
    return src.width == g_pending.width && src.height == g_pending.height;
}

// Native AA: there is no separate upscale pass, the tone mapper writes straight into the back buffer. Draw it,
// then sharpen the back buffer in place (before the UI is drawn on top).
static bool SharpenAfterToneMapper(command_list* cmd, bool indexed, uint32_t count, uint32_t instances,
                                   uint32_t first, int32_t vertexOffset, uint32_t firstInstance)
{
    auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(cmd->get_native());
    device* dev = cmd->get_device();
    const ViewInfo rt = GetViewInfo(dev, g_rtv0);
    const resource back = {rt.res};
    const resource_desc bd = dev->get_resource_desc(back);
    const format fmt = format_to_default_typed(bd.texture.format);
    if (indexed)
        ctx->DrawIndexedInstanced(count, instances, first, vertexOffset, firstInstance);
    else
        ctx->DrawInstanced(count, instances, first, firstInstance);
    ScopedOwnState own(ctx);
    if (!own.ok() || !EnsureSharpTarget(dev, bd.texture.width, bd.texture.height, fmt, true))
        return true;
    cmd->copy_resource(back, g_pass.copy);
    if (DispatchRcas(cmd, g_pass.copySrv, bd.texture.width, bd.texture.height))
        cmd->copy_resource(g_pass.sharp, back);
    return true;
}

static unsigned g_drawIndex = 0;  // draw calls so far in this frame

static void LogCandidate(device* dev)
{
    char srv[256] = "";
    for (int i = 0; i < 8; ++i)
        if (g_psSrv[i]) {
            const ViewInfo v = GetViewInfo(dev, g_psSrv[i]);
            char one[48];
            snprintf(one, sizeof(one), " t%d=%llx:%ux%u/f%u", i, (unsigned long long)(v.res & 0xffffff), v.width, v.height, (unsigned)v.fmt);
            strncat_s(srv, one, _TRUNCATE);
        }
    const ViewInfo rt = GetViewInfo(dev, g_rtv0);
    LOGI("candidate frame %u draw %u ps %016llx rt=%llx ds=%llx%s", g_frameId, g_drawIndex, (unsigned long long)PsHash(g_ps),
         (unsigned long long)(rt.res & 0xffffff), (unsigned long long)(GetViewInfo(dev, g_dsv).res & 0xffffff), srv);
}

// The final upscale pass of an upscaling frame: replaced once, then the frame needs nothing more.
static bool MaybeUpscaleFinal(command_list* cmd, bool indexed, uint32_t n, uint32_t instances, uint32_t first,
                              int32_t vertexOffset, uint32_t firstInstance)
{
    if (!IsFinalUpscaleDraw(cmd, n))
        return false;
    g_pending.frame = 0;
    if (!g_pending.upscale)
        return SharpenAfterToneMapper(cmd, indexed, n, instances, first, vertexOffset, firstInstance);
    return ReplaceFinalPass(cmd, indexed, n, instances, first, vertexOffset, firstInstance);
}

static bool OnDraw(command_list* cmd, uint32_t n, uint32_t instances, uint32_t first, uint32_t firstInstance)
{
    return MaybeUpscaleFinal(cmd, false, n, instances, first, 0, firstInstance);
}

static bool OnDrawIndexed(command_list* cmd, uint32_t n, uint32_t instances, uint32_t first, int32_t vertexOffset,
                          uint32_t firstInstance)
{
    ++g_drawIndex;
    if (MaybeUpscaleFinal(cmd, true, n, instances, first, vertexOffset, firstInstance))
        return true;
    if (n != 3)
        return false;
    RefreshDrawState(cmd);
    if (g_logActive && IsTaaDraw(cmd->get_device(), n))
        LogCandidate(cmd->get_device());
    if (g_replacedFrame == g_frameId)
        return IsTaaDraw(cmd->get_device(), n);  // second TAA draw (responsive pixels): already done
    if (!g_enabled || g_checkedFrame == g_frameId || !IsTaaDraw(cmd->get_device(), n))
        return false;
    g_checkedFrame = g_frameId;
    if (!ReplaceTaa(cmd))
        return false;
    g_replacedFrame = g_frameId;
    ++g_replacements;
    return true;
}

// ---------------------------------------------------------------- comparison screenshots
// Creating or rewriting Styx2Compare.flag next to the game exe takes three screenshots of the same view
// (stand still): upscaler + sharpening, upscaler without sharpening, the game's own TAA. Settings are restored
// afterwards. PNGs land next to the exe as Styx2Compare_<n>_<what>.png.

// Copies the back buffer to the CPU and writes it as a 32-bit PNG (WIC).
static bool SaveBackBufferPng(swapchain* sc, const std::wstring& path)
{
    auto* tex = reinterpret_cast<ID3D11Texture2D*>(sc->get_current_back_buffer().handle);
    auto* d3d = reinterpret_cast<ID3D11Device*>(sc->get_device()->get_native());
    D3D11_TEXTURE2D_DESC desc;
    tex->GetDesc(&desc);
    if (desc.Format != DXGI_FORMAT_R10G10B10A2_UNORM && desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM &&
        desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM) {
        LOGW("compare: back buffer format %d not supported", static_cast<int>(desc.Format));
        return false;
    }
    D3D11_TEXTURE2D_DESC sd = desc;
    sd.Usage = D3D11_USAGE_STAGING;
    sd.BindFlags = 0;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    sd.MiscFlags = 0;
    sd.SampleDesc = {1, 0};
    ID3D11Texture2D* staging = nullptr;
    if (FAILED(d3d->CreateTexture2D(&sd, nullptr, &staging)))
        return false;
    ID3D11DeviceContext* ctx = nullptr;
    d3d->GetImmediateContext(&ctx);
    ctx->CopyResource(staging, tex);
    std::vector<uint8_t> bgra(static_cast<size_t>(desc.Width) * desc.Height * 4);
    D3D11_MAPPED_SUBRESOURCE m;
    bool ok = SUCCEEDED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m));
    if (ok) {
        for (UINT y = 0; y < desc.Height; ++y) {
            const auto* src = static_cast<const uint8_t*>(m.pData) + static_cast<size_t>(y) * m.RowPitch;
            uint8_t* dst = bgra.data() + static_cast<size_t>(y) * desc.Width * 4;
            for (UINT x = 0; x < desc.Width; ++x, dst += 4) {
                uint32_t v;
                memcpy(&v, src + x * 4, 4);
                if (desc.Format == DXGI_FORMAT_R10G10B10A2_UNORM) {
                    dst[2] = static_cast<uint8_t>(((v >> 0) & 1023) * 255 / 1023);
                    dst[1] = static_cast<uint8_t>(((v >> 10) & 1023) * 255 / 1023);
                    dst[0] = static_cast<uint8_t>(((v >> 20) & 1023) * 255 / 1023);
                } else if (desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM) {
                    dst[2] = v & 255;
                    dst[1] = (v >> 8) & 255;
                    dst[0] = (v >> 16) & 255;
                } else {
                    dst[0] = v & 255;
                    dst[1] = (v >> 8) & 255;
                    dst[2] = (v >> 16) & 255;
                }
                dst[3] = 255;
            }
        }
        ctx->Unmap(staging, 0);
    }
    staging->Release();
    ctx->Release();
    if (!ok)
        return false;

    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IWICImagingFactory* factory = nullptr;
    IWICStream* stream = nullptr;
    IWICBitmapEncoder* encoder = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
    ok = SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))) &&
         SUCCEEDED(factory->CreateStream(&stream)) &&
         SUCCEEDED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE)) &&
         SUCCEEDED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder)) &&
         SUCCEEDED(encoder->Initialize(stream, WICBitmapEncoderNoCache)) &&
         SUCCEEDED(encoder->CreateNewFrame(&frame, nullptr)) && SUCCEEDED(frame->Initialize(nullptr)) &&
         SUCCEEDED(frame->SetSize(desc.Width, desc.Height)) && SUCCEEDED(frame->SetPixelFormat(&fmt)) &&
         SUCCEEDED(frame->WritePixels(desc.Height, desc.Width * 4, static_cast<UINT>(bgra.size()), bgra.data())) &&
         SUCCEEDED(frame->Commit()) && SUCCEEDED(encoder->Commit());
    if (frame) frame->Release();
    if (encoder) encoder->Release();
    if (stream) stream->Release();
    if (factory) factory->Release();
    return ok;
}

static void CompareFrameEnd(swapchain* sc)
{
    static ULONGLONG lastCheck = 0;
    static FILETIME lastFlag = {};
    static int step = -1, wait = 0, shot = 0;
    static bool savedEnabled = true, savedSharpen = true;
    struct Step {
        bool enabled, sharpen;
        int settleFrames;  // the temporal history needs a moment after a switch
        const char* name;
    };
    static const Step kSteps[] = {{true, true, 30, "1_upscaler_rcas"}, {true, false, 30, "2_upscaler"},
                                  {false, false, 120, "3_game_taa"}};
    if (step < 0) {
        if (GetTickCount64() - lastCheck < 1000)
            return;
        lastCheck = GetTickCount64();
        WIN32_FILE_ATTRIBUTE_DATA fa;
        char flag[MAX_PATH];
        GetModuleFileNameA(nullptr, flag, MAX_PATH);
        if (char* slash = strrchr(flag, '\\'))
            strcpy_s(slash + 1, MAX_PATH - (slash + 1 - flag), "Styx2Compare.flag");
        if (!GetFileAttributesExA(flag, GetFileExInfoStandard, &fa) || CompareFileTime(&fa.ftLastWriteTime, &lastFlag) == 0)
            return;
        const bool first = !lastFlag.dwLowDateTime && !lastFlag.dwHighDateTime;
        lastFlag = fa.ftLastWriteTime;
        if (first)
            return;  // flag left from an earlier session
        savedEnabled = g_enabled;
        savedSharpen = g_sharpenEnabled;
        step = 0;
        wait = kSteps[0].settleFrames;
        ++shot;
        g_enabled = kSteps[0].enabled;
        g_sharpenEnabled = kSteps[0].sharpen;
        LOGI("compare: taking screenshots, stand still");
        return;
    }
    if (--wait > 0)
        return;
    wchar_t dir[MAX_PATH];
    GetModuleFileNameW(nullptr, dir, MAX_PATH);
    if (wchar_t* slash = wcsrchr(dir, L'\\'))
        slash[1] = L'\0';
    wchar_t name[96];
    swprintf_s(name, L"Styx2Compare_%d_%hs.png", shot, kSteps[step].name);
    const bool saved = SaveBackBufferPng(sc, std::wstring(dir) + name);
    LOGI("compare: %s %s", kSteps[step].name, saved ? "saved" : "FAILED");
    if (++step < static_cast<int>(std::size(kSteps))) {
        g_enabled = kSteps[step].enabled;
        g_sharpenEnabled = kSteps[step].sharpen;
        g_up.resetNext = true;
        wait = kSteps[step].settleFrames;
        return;
    }
    g_enabled = savedEnabled;
    g_sharpenEnabled = savedSharpen;
    g_up.resetNext = true;
    step = -1;
    LOGI("compare: done");
}

static void OnPresent(command_queue*, swapchain* sc, const rect*, const rect*, uint32_t, const rect*)
{
    ++g_frameId;
    LatencyFrameEnd();
    CompareFrameEnd(sc);
    g_drawIndex = 0;
    // [Debug] LogCandidates=N logs N frames, 20 s after start-up (main menu) and again after 60 s.
    static ULONGLONG start = GetTickCount64();
    static int rounds = 0, budget = 0;
    const ULONGLONG t = GetTickCount64() - start;
    if (g_logCandidates > 0 && budget == 0 && ((rounds == 0 && t > 20000) || (rounds == 1 && t > 60000))) {
        budget = g_logCandidates;
        ++rounds;
    }
    g_logActive = budget > 0;
    if (budget > 0)
        --budget;
    static ULONGLONG lastLog = GetTickCount64();
    static unsigned frames = 0;
    ++frames;
    if (GetTickCount64() - lastLog >= 10000) {
        lastLog = GetTickCount64();
        LOGI("upscaler: %u of %u frames replaced TAA, %u evaluations, %u failures", g_replacements, frames,
             g_up.evaluations, g_up.failures);
        g_replacements = frames = 0;
        g_up.evaluations = 0;
    }
}

static void OnDestroyDevice(device* dev)
{
    ShutdownLatency(dev);
    ReleaseOwnState();
    ShutdownUpscaler(dev);
    DestroyPassTextures(dev);
    if (g_pass.pipe.handle) dev->destroy_pipeline(g_pass.pipe);
    if (g_pass.layout.handle) dev->destroy_pipeline_layout(g_pass.layout);
    if (g_pass.sanitize.handle) dev->destroy_pipeline(g_pass.sanitize);
    if (g_pass.rcas.handle) dev->destroy_pipeline(g_pass.rcas);
    if (g_pass.rcasLayout.handle) dev->destroy_pipeline_layout(g_pass.rcasLayout);
    if (g_pass.sanitizeLayout.handle) dev->destroy_pipeline_layout(g_pass.sanitizeLayout);
    g_pass = {};
}

// ---------------------------------------------------------------- quality presets / engine settings
// UE 4.13 reads its console variables at start-up, so the add-on keeps the user Engine.ini in shape: TAA on
// (the game ships with FXAA, which has no jitter or history for the upscaler) and the render resolution of
// the chosen preset. Changes apply on the next start.

struct QualityPreset {
    const char* name;
    int percent;
};
static const QualityPreset kPresets[] = {{"Native AA", 100}, {"Quality", 67}, {"Balanced", 59}, {"Performance", 50}};
static int g_quality = 0;          // index into kPresets
static int g_qualityAtStart = 0;   // preset the game was started with

static std::string AddonIniPath()
{
    char path[MAX_PATH];
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (char* slash = strrchr(path, '\\'))
        slash[1] = '\0';
    return std::string(path) + "Styx2Upscaler.ini";
}

static std::string EngineIniPath()
{
    char local[MAX_PATH];
    if (!GetEnvironmentVariableA("LOCALAPPDATA", local, MAX_PATH))
        return {};
    return std::string(local) + "\\Styx2\\Saved\\Config\\WindowsNoEditor\\Engine.ini";
}

static void SetEngineValue(const std::string& ini, const char* section, const char* key, const char* value)
{
    char current[32];
    GetPrivateProfileStringA(section, key, "", current, sizeof(current), ini.c_str());
    if (strcmp(current, value) != 0)
        WritePrivateProfileStringA(section, key, value, ini.c_str());
}

static bool g_manageEngineIni = true;  // [Upscaler] Enabled at start: 0 leaves Engine.ini to the game (FXAA, 100%)

static void ApplyEngineSettings()
{
    const std::string ini = EngineIniPath();
    if (ini.empty() || !g_manageEngineIni)
        return;
    // Before the game's very first start the folder does not exist yet; the game keeps entries it finds.
    for (size_t pos = ini.find('\\', 3); pos != std::string::npos; pos = ini.find('\\', pos + 1))
        CreateDirectoryA(ini.substr(0, pos).c_str(), nullptr);
    char percent[8];
    snprintf(percent, sizeof(percent), "%d", kPresets[g_quality].percent);
    SetEngineValue(ini, "SystemSettings", "r.DefaultFeature.AntiAliasing", "2");
    SetEngineValue(ini, "SystemSettings", "r.PostProcessAAQuality", "4");
    SetEngineValue(ini, "SystemSettings", "r.ScreenPercentage", percent);
    SetEngineValue(ini, "/Script/Engine.RendererSettings", "r.DefaultFeature.AntiAliasing", "2");
}

// OptiScaler's own overlay needs it to start with the game, which breaks this game's rendering, so the upscaler
// it runs is chosen here and written to OptiScaler.ini (read when OptiScaler is loaded: applies on next start).
struct UpscalerChoice {
    const char* name;
    const char* value;  // OptiScaler.ini [Upscalers] Dx11Upscaler
};
static const UpscalerChoice kUpscalers[] = {
    {"FSR 4 (RX 9000) / FSR 3.1", "fsr31_12"},
    {"XeSS", "xess_12"},
    {"DLSS (NVIDIA RTX, needs nvngx_dlss.dll)", "dlss"},
    {"FSR 2.2 (native DX11)", "fsr22"},
};
static int g_upscalerChoice = -1, g_upscalerChoiceAtStart = -1;  // -1 = not one of the above

static std::string OptiScalerIniPath()
{
    std::string ini = AddonIniPath();
    return ini.substr(0, ini.find_last_of('\\') + 1) + "OptiScaler.ini";
}

// The choice lives in Styx2Upscaler.ini ([Upscaler] Backend) and is copied into OptiScaler.ini right before the
// add-on loads OptiScaler: OptiScaler rewrites its own ini from memory when the game exits.
static void ReadUpscalerChoice()
{
    char value[32];
    GetPrivateProfileStringA("Upscaler", "Backend", "", value, sizeof(value), AddonIniPath().c_str());
    if (!value[0])
        GetPrivateProfileStringA("Upscalers", "Dx11Upscaler", "", value, sizeof(value), OptiScalerIniPath().c_str());
    for (int i = 0; i < static_cast<int>(std::size(kUpscalers)); ++i)
        if (_stricmp(value, kUpscalers[i].value) == 0)
            g_upscalerChoice = i;
    g_upscalerChoiceAtStart = g_upscalerChoice;
}

static void WriteUpscalerChoiceToOptiScaler()
{
    if (g_upscalerChoice >= 0)
        WritePrivateProfileStringA("Upscalers", "Dx11Upscaler", kUpscalers[g_upscalerChoice].value, OptiScalerIniPath().c_str());
}

// ---------------------------------------------------------------- settings overlay

static void DrawSettings(effect_runtime*)
{
    if (ImGui::Checkbox("Upscaler (FSR / DLSS / XeSS via OptiScaler) instead of TAA", &g_enabled))
        g_up.resetNext = true;
    ImGui::TextDisabled("Off = the game's own TAA, for comparison.");
    if (ImGui::BeginCombo("Upscaler", g_upscalerChoice >= 0 ? kUpscalers[g_upscalerChoice].name : "Custom (OptiScaler.ini)")) {
        for (int i = 0; i < static_cast<int>(std::size(kUpscalers)); ++i)
            if (ImGui::Selectable(kUpscalers[i].name, i == g_upscalerChoice)) {
                g_upscalerChoice = i;
                WritePrivateProfileStringA("Upscaler", "Backend", kUpscalers[i].value, AddonIniPath().c_str());
            }
        ImGui::EndCombo();
    }
    char preview[48];
    snprintf(preview, sizeof(preview), "%s (%d%%)", kPresets[g_quality].name, kPresets[g_quality].percent);
    if (ImGui::BeginCombo("Quality", preview)) {
        for (int i = 0; i < static_cast<int>(std::size(kPresets)); ++i) {
            char label[48];
            snprintf(label, sizeof(label), "%s (%d%%)", kPresets[i].name, kPresets[i].percent);
            if (ImGui::Selectable(label, i == g_quality)) {
                g_quality = i;
                char v[8];
                snprintf(v, sizeof(v), "%d", i);
                WritePrivateProfileStringA("Upscaler", "Quality", v, AddonIniPath().c_str());
                ApplyEngineSettings();
            }
        }
        ImGui::EndCombo();
    }
    if (g_quality != g_qualityAtStart || g_upscalerChoice != g_upscalerChoiceAtStart)
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Restart the game to apply the change.");
    else
        ImGui::TextDisabled("Native AA = best image. Lower presets help when the GPU is the bottleneck.");
    // Live: the final pass reads both every frame. The strength is saved when the slider is released.
    if (ImGui::Checkbox("Sharpening (RCAS)", &g_sharpenEnabled))
        WritePrivateProfileStringA("Upscaler", "Sharpening", g_sharpenEnabled ? "1" : "0", AddonIniPath().c_str());
    ImGui::BeginDisabled(!g_sharpenEnabled);
    ImGui::SliderFloat("Strength", &g_sharpness, 0.0f, 2.0f, "%.2f");
    ImGui::EndDisabled();
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        char value[16];
        snprintf(value, sizeof(value), "%.2f", g_sharpness);
        WritePrivateProfileStringA("Upscaler", "Sharpness", value, AddonIniPath().c_str());
    }
    ImGui::TextDisabled("AMD FidelityFX RCAS on the final image. 1 = full RCAS, 2 = strongest.");
    ImGui::Separator();
    if (ImGui::Checkbox("AMD Anti-Lag 2", &g_antiLag2Wanted))
        WritePrivateProfileStringA("Latency", "AntiLag2", g_antiLag2Wanted ? "1" : "0", AddonIniPath().c_str());
    if (g_latencyMode == kLatencyAntiLag2)
        ImGui::TextDisabled(g_antiLag2Wanted ? "Active (Radeon driver)" : "Off");
    else if (g_latencyMode == kLatencyFallback)
        ImGui::TextDisabled("Not available on this GPU/driver: frame queue limited to %d instead", g_maxFrameLatency);
    ImGui::Separator();
    ImGui::Combo("Debug view", &g_debugView, "Off\0Motion vectors\0Pass-through (no upscaler)\0Velocity check\0");
    if (g_up.width)
        ImGui::TextDisabled("Rendering %ux%u, output %ux%u, %u failures", g_up.renderWidth, g_up.renderHeight, g_up.width,
                            g_up.height, g_up.failures);
}

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID)
{
    switch (reason) {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(module))
            return FALSE;
        {
            // Styx2Upscaler.ini next to the exe: [Upscaler] Enabled, Quality (preset index); [Debug] View
            // (0 off, 1 motion vectors, 2 pass-through, 3 velocity check), LogCandidates
            const std::string path = AddonIniPath();
            g_enabled = GetPrivateProfileIntA("Upscaler", "Enabled", 1, path.c_str()) != 0;
            g_manageEngineIni = g_enabled;
            g_quality = GetPrivateProfileIntA("Upscaler", "Quality", 0, path.c_str());
            if (g_quality < 0 || g_quality >= static_cast<int>(std::size(kPresets)))
                g_quality = 0;
            g_qualityAtStart = g_quality;
            g_antiLag2Wanted = GetPrivateProfileIntA("Latency", "AntiLag2", 1, path.c_str()) != 0;
            char sharpness[16];
            g_sharpenEnabled = GetPrivateProfileIntA("Upscaler", "Sharpening", 1, path.c_str()) != 0;
            GetPrivateProfileStringA("Upscaler", "Sharpness", "1.0", sharpness, sizeof(sharpness), path.c_str());
            g_sharpness = std::clamp(static_cast<float>(atof(sharpness)), 0.0f, 2.0f);
            g_maxFrameLatency = GetPrivateProfileIntA("Latency", "MaxFrameLatency", 1, path.c_str());
            g_debugView = GetPrivateProfileIntA("Debug", "View", 0, path.c_str());
            g_logCandidates = GetPrivateProfileIntA("Debug", "LogCandidates", 0, path.c_str());
            ApplyEngineSettings();
            ReadUpscalerChoice();
        }
        reshade::register_overlay(nullptr, &DrawSettings);
        reshade::register_event<reshade::addon_event::init_resource>(OnInitResource);
        reshade::register_event<reshade::addon_event::destroy_resource>(OnDestroyResource);
        reshade::register_event<reshade::addon_event::destroy_resource_view>(OnDestroyResourceView);
        reshade::register_event<reshade::addon_event::map_buffer_region>(OnMapBuffer);
        reshade::register_event<reshade::addon_event::unmap_buffer_region>(OnUnmapBuffer);
        reshade::register_event<reshade::addon_event::update_buffer_region>(OnUpdateBuffer);
        reshade::register_event<reshade::addon_event::init_pipeline>(OnInitPipeline);
        reshade::register_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
        reshade::register_event<reshade::addon_event::draw>(OnDraw);
        reshade::register_event<reshade::addon_event::present>(OnPresent);
        reshade::register_event<reshade::addon_event::init_swapchain>(OnInitSwapchain);        reshade::register_event<reshade::addon_event::destroy_device>(OnDestroyDevice);
        LOGI("Styx 2 Upscaler loaded");
        break;
    case DLL_PROCESS_DETACH:
        // The game writes Engine.ini back from memory when it exits, dropping entries added after it read the
        // file: write them again now, after that, so the next start has TAA and the chosen render resolution.
        ApplyEngineSettings();
        reshade::unregister_addon(module);
        break;
    }
    return TRUE;
}
