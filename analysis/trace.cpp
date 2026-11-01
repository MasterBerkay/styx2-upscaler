// Styx2Trace - ReShade add-on that records two frames of Styx: Shards of Darkness (UE 4.13, DX11).
// Creating Styx2Trace.flag next to the game exe (no hotkey: the game uses most keys) writes Styx2Trace.txt:
// every pass with its render targets, shader resources and the reflected resource names of its shaders,
// to find the TAA pass and its inputs.

#include <reshade.hpp>
#include <d3d11.h>
#include <d3d11shader.h>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace reshade::api;

extern "C" __declspec(dllexport) const char* NAME = "Styx2Trace";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Frame trace for Styx: Shards of Darkness (analysis only).";

using PFN_D3DReflect = HRESULT(WINAPI*)(const void*, SIZE_T, REFIID, void**);

static std::mutex g_mutex;
static std::unordered_map<uint64_t, std::string> g_shaderInfo;  // shader handle -> reflected names
static std::unordered_map<uint64_t, int> g_shaderId;

static PFN_D3DReflect LoadReflect()
{
    static PFN_D3DReflect fn = [] {
        HMODULE m = LoadLibraryA("d3dcompiler_47.dll");
        return m ? reinterpret_cast<PFN_D3DReflect>(GetProcAddress(m, "D3DReflect")) : nullptr;
    }();
    return fn;
}

// "t0:SceneColor t1:Velocity b0:View{TemporalAAParams@..}" for one shader.
static std::string Reflect(const void* code, size_t size)
{
    std::string out;
    PFN_D3DReflect reflect = LoadReflect();
    ID3D11ShaderReflection* refl = nullptr;
    if (!reflect || FAILED(reflect(code, size, IID_ID3D11ShaderReflection, reinterpret_cast<void**>(&refl))))
        return "(no reflection)";
    D3D11_SHADER_DESC sd;
    refl->GetDesc(&sd);
    char buf[256];
    for (UINT i = 0; i < sd.BoundResources; ++i) {
        D3D11_SHADER_INPUT_BIND_DESC bd;
        if (FAILED(refl->GetResourceBindingDesc(i, &bd)))
            continue;
        const char kind = bd.Type == D3D_SIT_CBUFFER ? 'b' : bd.Type == D3D_SIT_SAMPLER ? 's'
                        : (bd.Type == D3D_SIT_UAV_RWTYPED || bd.Type == D3D_SIT_UAV_RWSTRUCTURED) ? 'u' : 't';
        if (kind == 's')
            continue;
        snprintf(buf, sizeof(buf), " %c%u:%s", kind, bd.BindPoint, bd.Name);
        out += buf;
        if (kind == 'b') {  // list the used variables of each constant buffer
            ID3D11ShaderReflectionConstantBuffer* cb = refl->GetConstantBufferByName(bd.Name);
            D3D11_SHADER_BUFFER_DESC cbd;
            if (cb && SUCCEEDED(cb->GetDesc(&cbd))) {
                out += "{";
                for (UINT v = 0; v < cbd.Variables; ++v) {
                    D3D11_SHADER_VARIABLE_DESC vd;
                    if (SUCCEEDED(cb->GetVariableByIndex(v)->GetDesc(&vd)) && (vd.uFlags & D3D_SVF_USED)) {
                        snprintf(buf, sizeof(buf), "%s@%u ", vd.Name, vd.StartOffset);
                        out += buf;
                    }
                }
                out += "}";
            }
        }
    }
    refl->Release();
    return out;
}

static void OnInitPipeline(device*, pipeline_layout, uint32_t count, const pipeline_subobject* subs, pipeline pipe)
{
    for (uint32_t i = 0; i < count; ++i) {
        const char* stage = subs[i].type == pipeline_subobject_type::vertex_shader ? "VS"
                          : subs[i].type == pipeline_subobject_type::pixel_shader ? "PS"
                          : subs[i].type == pipeline_subobject_type::compute_shader ? "CS" : nullptr;
        if (!stage || !subs[i].data)
            continue;
        if (subs[i].type == pipeline_subobject_type::vertex_shader && count > 1)
            continue;  // input layout creation also reports the VS bytecode
        const auto* d = static_cast<const shader_desc*>(subs[i].data);
        std::string info = std::string(stage) + Reflect(d->code, d->code_size);
        std::lock_guard<std::mutex> lock(g_mutex);
        g_shaderInfo[pipe.handle] = info;
        g_shaderId.erase(pipe.handle);
    }
}

// ---------------------------------------------------------------- trace

static bool g_tracing = false;
static int g_traceFrames = 0;
static std::vector<std::string> g_lines;
static std::unordered_map<uint64_t, int> g_resId;
static uint64_t g_vs = 0, g_ps = 0, g_cs = 0;
static std::string g_rt;  // current render targets, formatted
static uint64_t g_psSrv[16] = {}, g_csSrv[16] = {}, g_csUav[8] = {};

static void Line(const char* fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_lines.emplace_back(buf);
}

static int ShaderId(uint64_t h)
{
    if (!h)
        return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_shaderId.find(h);
    if (it != g_shaderId.end())
        return it->second;
    int id = static_cast<int>(g_shaderId.size()) + 1;
    g_shaderId[h] = id;
    return id;
}

// "R12 1280x720 r16g16b16a16_float" for the resource behind a view.
static std::string ViewText(device* dev, uint64_t view)
{
    if (!view)
        return "-";
    resource r = dev->get_resource_from_view(resource_view{view});
    if (!r.handle)
        return "?";
    auto it = g_resId.find(r.handle);
    int id = it != g_resId.end() ? it->second : (g_resId[r.handle] = static_cast<int>(g_resId.size()) + 1);
    resource_desc d = dev->get_resource_desc(r);
    char buf[96];
    if (d.type == resource_type::buffer)
        snprintf(buf, sizeof(buf), "R%d buf%llu", id, (unsigned long long)d.buffer.size);
    else
        snprintf(buf, sizeof(buf), "R%d %ux%u f%u%s", id, d.texture.width, d.texture.height, (unsigned)d.texture.format,
                 d.texture.levels > 1 ? " mips" : "");
    return buf;
}

static std::string SrvList(device* dev, const uint64_t* srvs, int n)
{
    std::string s;
    for (int i = 0; i < n; ++i)
        if (srvs[i])
            s += " t" + std::to_string(i) + "=" + ViewText(dev, srvs[i]);
    return s;
}

static void OnBindRenderTargets(command_list* cmd, uint32_t count, const resource_view* rtvs, resource_view dsv)
{
    if (!g_tracing)
        return;
    device* dev = cmd->get_device();
    g_rt.clear();
    for (uint32_t i = 0; i < count; ++i)
        g_rt += "rt" + std::to_string(i) + "=" + ViewText(dev, rtvs[i].handle) + " ";
    g_rt += "ds=" + ViewText(dev, dsv.handle);
}

static void OnBindPipeline(command_list*, pipeline_stage stages, pipeline pipe)
{
    if ((stages & pipeline_stage::vertex_shader) == pipeline_stage::vertex_shader) g_vs = pipe.handle;
    if ((stages & pipeline_stage::pixel_shader) == pipeline_stage::pixel_shader) g_ps = pipe.handle;
    if ((stages & pipeline_stage::compute_shader) == pipeline_stage::compute_shader) g_cs = pipe.handle;
}

static void OnPushDescriptors(command_list*, shader_stage stages, pipeline_layout, uint32_t, const descriptor_table_update& u)
{
    uint64_t* table = nullptr;
    int size = 0;
    if (u.type == descriptor_type::shader_resource_view) {
        if ((stages & shader_stage::pixel) == shader_stage::pixel) { table = g_psSrv; size = 16; }
        else if ((stages & shader_stage::compute) == shader_stage::compute) { table = g_csSrv; size = 16; }
    } else if (u.type == descriptor_type::unordered_access_view && (stages & shader_stage::compute) == shader_stage::compute) {
        table = g_csUav; size = 8;
    }
    if (!table)
        return;
    const auto* views = static_cast<const resource_view*>(u.descriptors);
    for (uint32_t i = 0; i < u.count; ++i)
        if (u.binding + i < static_cast<uint32_t>(size))
            table[u.binding + i] = views[i].handle;
}

static bool OnDraw(command_list* cmd, uint32_t n, uint32_t inst, uint32_t, uint32_t)
{
    if (g_tracing)
        Line("DRAW n=%u x%u vs=%d ps=%d | %s |%s", n, inst, ShaderId(g_vs), ShaderId(g_ps), g_rt.c_str(),
             SrvList(cmd->get_device(), g_psSrv, 16).c_str());
    return false;
}

// The TAA pass: full-screen triangle reading depth (t0) and the R16G16 velocity buffer (t3).
static bool IsTaaDraw(device* dev, uint32_t n)
{
    if (n != 3 || !g_psSrv[0] || !g_psSrv[3])
        return false;
    const resource_desc d0 = dev->get_resource_desc(dev->get_resource_from_view(resource_view{g_psSrv[0]}));
    const resource_desc d3 = dev->get_resource_desc(dev->get_resource_from_view(resource_view{g_psSrv[3]}));
    return d3.texture.format == format::r16g16_unorm && d0.texture.format == format::r24_g8_typeless;
}

// Copies every constant buffer bound to the pixel shader to a staging buffer and logs it as floats.
static void DumpPsConstants(command_list* cmd)
{
    auto* ctx = reinterpret_cast<ID3D11DeviceContext*>(cmd->get_native());
    ID3D11Device* d3d = nullptr;
    ctx->GetDevice(&d3d);
    ID3D11Buffer* cbs[14] = {};
    ctx->PSGetConstantBuffers(0, 14, cbs);
    for (int slot = 0; slot < 14; ++slot) {
        if (!cbs[slot])
            continue;
        D3D11_BUFFER_DESC bd;
        cbs[slot]->GetDesc(&bd);
        D3D11_BUFFER_DESC sd = {};
        sd.ByteWidth = bd.ByteWidth;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Buffer* staging = nullptr;
        if (SUCCEEDED(d3d->CreateBuffer(&sd, nullptr, &staging))) {
            ctx->CopyResource(staging, cbs[slot]);
            D3D11_MAPPED_SUBRESOURCE m;
            if (SUCCEEDED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m))) {
                Line("  CB b%d size %u", slot, bd.ByteWidth);
                const float* f = static_cast<const float*>(m.pData);
                for (UINT o = 0; o < bd.ByteWidth / 4; o += 4)
                    Line("    @%4u  %12.6g %12.6g %12.6g %12.6g", o * 4, f[o], f[o + 1], f[o + 2], f[o + 3]);
                ctx->Unmap(staging, 0);
            }
            staging->Release();
        }
        cbs[slot]->Release();
    }
    d3d->Release();
}

static bool OnDrawIndexed(command_list* cmd, uint32_t n, uint32_t inst, uint32_t, int32_t, uint32_t)
{
    if (!g_tracing)
        return false;
    Line("DRAWIDX n=%u x%u vs=%d ps=%d | %s |%s", n, inst, ShaderId(g_vs), ShaderId(g_ps), g_rt.c_str(),
         SrvList(cmd->get_device(), g_psSrv, 16).c_str());
    static uint64_t dumpedPs = 0;
    static int dumpedFrame = -1;
    if (IsTaaDraw(cmd->get_device(), n) && (dumpedFrame != g_traceFrames || dumpedPs != g_ps)) {
        dumpedFrame = g_traceFrames;
        dumpedPs = g_ps;
        Line("  TAA constants (frame %d):", g_traceFrames);
        DumpPsConstants(cmd);
    }
    return false;
}

static bool OnDispatch(command_list* cmd, uint32_t x, uint32_t y, uint32_t z)
{
    if (!g_tracing)
        return false;
    device* dev = cmd->get_device();
    std::string uav;
    for (int i = 0; i < 8; ++i)
        if (g_csUav[i])
            uav += " u" + std::to_string(i) + "=" + ViewText(dev, g_csUav[i]);
    Line("DISPATCH %ux%ux%u cs=%d |%s |%s", x, y, z, ShaderId(g_cs), uav.c_str(), SrvList(dev, g_csSrv, 16).c_str());
    return false;
}

static bool OnClearRtv(command_list* cmd, resource_view rtv, const float*, uint32_t, const rect*)
{
    if (g_tracing)
        Line("CLEAR_RT %s", ViewText(cmd->get_device(), rtv.handle).c_str());
    return false;
}

static bool OnClearDsv(command_list* cmd, resource_view dsv, const float*, const uint8_t*, uint32_t, const rect*)
{
    if (g_tracing)
        Line("CLEAR_DS %s", ViewText(cmd->get_device(), dsv.handle).c_str());
    return false;
}

static bool OnCopyResource(command_list* cmd, resource src, resource dst)
{
    if (!g_tracing)
        return false;
    device* dev = cmd->get_device();
    auto id = [&](resource r) {
        auto it = g_resId.find(r.handle);
        return it != g_resId.end() ? it->second : (g_resId[r.handle] = static_cast<int>(g_resId.size()) + 1);
    };
    resource_desc d = dev->get_resource_desc(src);
    Line("COPY R%d -> R%d (%ux%u f%u)", id(src), id(dst), d.texture.width, d.texture.height, (unsigned)d.texture.format);
    return false;
}

static std::string GameDirFile(const char* name)
{
    char path[MAX_PATH];
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    char* slash = strrchr(path, '\\');
    if (slash)
        slash[1] = '\0';
    return std::string(path) + name;
}

static void OnPresent(command_queue*, swapchain*, const rect*, const rect*, uint32_t, const rect*)
{
    // A new or rewritten flag file starts a trace (checked once a second).
    static ULONGLONG lastCheck = 0;
    static FILETIME lastFlag = {};
    bool start = false;
    if (!g_tracing && GetTickCount64() - lastCheck > 1000) {
        lastCheck = GetTickCount64();
        WIN32_FILE_ATTRIBUTE_DATA fa;
        if (GetFileAttributesExA(GameDirFile("Styx2Trace.flag").c_str(), GetFileExInfoStandard, &fa) &&
            CompareFileTime(&fa.ftLastWriteTime, &lastFlag) != 0) {
            start = lastFlag.dwLowDateTime || lastFlag.dwHighDateTime;  // not on the first sighting at start-up
            lastFlag = fa.ftLastWriteTime;
        }
    }
    if (g_tracing) {
        Line("---- PRESENT ----");
        if (++g_traceFrames == 2) {
            g_tracing = false;
            FILE* f = nullptr;
            if (fopen_s(&f, GameDirFile("Styx2Trace.txt").c_str(), "w") == 0 && f) {
                for (const auto& l : g_lines)
                    fprintf(f, "%s\n", l.c_str());
                fprintf(f, "\n==== SHADERS ====\n");
                std::lock_guard<std::mutex> lock(g_mutex);
                for (const auto& [h, id] : g_shaderId) {
                    auto it = g_shaderInfo.find(h);
                    fprintf(f, "%d: %s\n", id, it != g_shaderInfo.end() ? it->second.c_str() : "(unknown)");
                }
                fclose(f);
            }
            reshade::log::message(reshade::log::level::info, "Styx2Trace: trace written");
        }
    } else if (start) {
        g_tracing = true;
        g_traceFrames = 0;
        g_lines.clear();
        g_resId.clear();
        reshade::log::message(reshade::log::level::info, "Styx2Trace: tracing two frames");
    }
}

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID)
{
    switch (reason) {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(module))
            return FALSE;
        reshade::register_event<reshade::addon_event::init_pipeline>(OnInitPipeline);
        reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(OnBindRenderTargets);
        reshade::register_event<reshade::addon_event::bind_pipeline>(OnBindPipeline);
        reshade::register_event<reshade::addon_event::push_descriptors>(OnPushDescriptors);
        reshade::register_event<reshade::addon_event::draw>(OnDraw);
        reshade::register_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
        reshade::register_event<reshade::addon_event::dispatch>(OnDispatch);
        reshade::register_event<reshade::addon_event::clear_render_target_view>(OnClearRtv);
        reshade::register_event<reshade::addon_event::clear_depth_stencil_view>(OnClearDsv);
        reshade::register_event<reshade::addon_event::copy_resource>(OnCopyResource);
        reshade::register_event<reshade::addon_event::present>(OnPresent);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_addon(module);
        break;
    }
    return TRUE;
}
