# Styx 2 Upscaler – FSR 4 / DLSS / XeSS for Styx: Shards of Darkness

Real temporal upscaling and anti-aliasing for *Styx: Shards of Darkness* (2017, Unreal Engine 4.13, DX11):
the game's TAA is replaced by **FSR 4** (RX 9000), **FSR 3.1**, **XeSS** or **DLSS** through
[OptiScaler](https://github.com/optiscaler/OptiScaler).

- Native AA at the TAA pass (HDR), or quality presets (67 / 59 / 50 %) upscaled at the game's final upscale pass
- Camera motion vectors from depth plus the engine's own velocity buffer, jitter from the View uniform buffer
- AMD Anti-Lag 2 (DX11 SDK) with a `SetMaximumFrameLatency` fallback on other GPUs
- Settings menu in the ReShade overlay (Dear ImGui); upscaler / preset choice applied on the next start
- Turns on the engine's TAA (`r.DefaultFeature.AntiAliasing=2`) in the user `Engine.ini`

Player downloads (bundled with ReShade and OptiScaler) are on Nexus Mods. Sister projects for *Styx: Master
of Shadows*: [styx-upscaler](https://github.com/MasterBerkay/styx-upscaler) and
[styx-engine-fix](https://github.com/MasterBerkay/styx-engine-fix).

## How it works

ReShade 6.x add-on, events on the immediate context. Renderer details and offsets are in `NOTES.md`.

1. **TAA detection** – the two TAA pixel shaders are recognised by an FNV-1a hash of their bytecode
   (reflection data is stripped in this build) plus their bindings: depth `R24G8` at t0, scene color at t1,
   history at t2, velocity `R16G16_UNORM` at t3, RGBA16F render target.
2. **View constants** – UE4 uploads `FViewUniformShaderParameters` (4096 bytes) by map / update / initial
   data; the add-on keeps a CPU copy of every buffer of that size and reads the one bound at PS b1:
   `ClipToTranslatedWorld`, `PrevTranslatedWorldToClip`, both `PreViewTranslation`s and both projection jitters.
3. **Motion vectors** – a compute shader reprojects depth (reversed Z) into the previous frame and replaces
   the result with the decoded velocity buffer where moving objects wrote one; both are made jitter-free.
   It also writes depth as R32F and a NaN/Inf-free copy of the scene color.
4. **Own device context state** – UE4's D3D11 RHI caches bound state and skips redundant sets. All add-on
   work runs inside its own `ID3DDeviceContextState` (`SwapDeviceContextState`), so the game's state is
   untouched – this also unbinds the game's depth-stencil so depth can be read.
5. **Native AA** – NGX evaluate (HDR, auto exposure), sanitize, copy into the TAA render target; both TAA
   draws are skipped.
6. **Upscaling** – below 100 % the scene buffers are allocated at render size and the game's final pass
   scales the tone-mapped image to the back buffer. The TAA pass then only passes the raw image on; at that
   final pass the tone-mapped image is upscaled (LDR) and the game's draw is issued again with the result in t0.
7. **OptiScaler as nvngx.dll** – loaded by the add-on on first use and called through its exported NGX
   functions (`NVSDK_NGX_D3D11_Init` is the 5-parameter form). As a proxy DLL loaded at start-up its device /
   DXGI hooks break this game's lighting. The upscaler choice is written to `OptiScaler.ini` right before loading.
8. **Latency** – AMD Anti-Lag 2 `Update()` at every present of the swap chain device.

`analysis/` holds the frame-trace add-on used to find all of this.

## Building

Visual Studio 2022 Build Tools (x64), then `build.bat` → `bin\Styx2Upscaler.addon64`.

Put the dependencies in `third_party\` (not part of this repository):

| Path | From |
|---|---|
| `third_party\reshade\include` | `include` folder of [ReShade 6.8.0](https://github.com/crosire/reshade) (BSD-3) |
| `third_party\ngx\include` | `include` folder of the [NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS) (types and parameter names only, no library) |

The NGX SDK is licensed by NVIDIA under the NVIDIA RTX SDKs License, which you accept by downloading it.

## Installing a build

Next to `Styx2\Binaries\Win64\Styx2-Win64-Shipping.exe`: ReShade 6.x **with add-on support** as `dxgi.dll`,
OptiScaler as **`nvngx.dll`** with `Dxgi=false` and `OverrideNvapiDll=false` in `OptiScaler.ini`, the FidelityFX /
XeSS DLLs and `D3D12_Optiscaler\` from the OptiScaler release, `Styx2Upscaler.addon64` and
`dist\Styx2Upscaler.ini`. Start the game twice (the first start writes the TAA setting to `Engine.ini`).

## License

MIT (see `LICENSE`) for the code in this repository. `src/ffx_antilag2_dx11.h` is the AMD Anti-Lag 2 SDK
(MIT, © Advanced Micro Devices, see `src/ffx_antilag2_LICENSE.txt`); `src/imgui/` holds the Dear ImGui 1.92.5
(docking) headers (MIT, © Omar Cornut, see `src/imgui/LICENSE.txt`). Styx: Shards of Darkness is © Cyanide /
Focus Home Interactive; no game files are included. Not affiliated with Cyanide, Focus, AMD, NVIDIA or Intel.
