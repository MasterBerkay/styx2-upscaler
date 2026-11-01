# Styx: Shards of Darkness - renderer notes (UE 4.13.1, DX11)

Findings from Styx2Trace (ReShade add-on, `src/trace.cpp`), 2026-11-01. Shader reflection is stripped,
so passes are recognised by structure.

## TAA must be forced
The game's "Anti-Aliasing Epic" is FXAA. TAA comes on with the user Engine.ini
(`%LOCALAPPDATA%\Styx2\Saved\Config\WindowsNoEditor\Engine.ini`):

```ini
[SystemSettings]
r.DefaultFeature.AntiAliasing=2
r.PostProcessAAQuality=4

[/Script/Engine.RendererSettings]
r.DefaultFeature.AntiAliasing=2
```

Edit it only while the game is closed: the game rewrites Engine.ini on exit.

## TAA pass
- One full-screen pass (ps "169") composites scene color + separate translucency into an RGBA16F target
  (TAA input color).
- TAA = two full-screen triangles (n=3), normal + responsive (stencil), same render target, depth bound:
  - t0 = scene depth (R24G8_TYPELESS, 2560x1440)
  - t1 = TAA input color (RGBA16F)
  - t2 = history (RGBA16F), ping-pongs with the output target every frame
  - t3 = velocity (R16G16_UNORM), only moving objects, 0 elsewhere
  - rt0 = TAA output (RGBA16F) -> bloom, tonemapper (-> BGRA8), final pass to back buffer (R10G10B10A2), UI
- Detection used by the trace: n == 3, t0 R24G8_TYPELESS, t3 R16G16_UNORM.

## Constant buffers at the TAA draw (pixel shader)
- b0 (1024 B): post-process globals; @0 PostprocessInput0Size (2560, 1440, 1/w, 1/h), @256 ScreenPosToPixel.
- b1 (4096 B): View uniform buffer, matches the UE 4.13 FViewUniformShaderParameters order:
  - @0    TranslatedWorldToClip (jittered)
  - @64   WorldToClip
  - @384  ViewToClip; row 2 (@416) = (jitterX, jitterY, 0, 1) in NDC
  - @448  ClipToView
  - @512  ClipToTranslatedWorld
  - @896  PreViewTranslation (float3)
  - @912  PrevProjection; row 2 (@944) = previous frame's jitter
  - @976  PrevViewProj
  - @1232 PrevTranslatedWorldToClip (jittered, previous frame)
  - @1584 PrevPreViewTranslation (float3)
  - @1728 looks like ClipToPrevClip but reads identity: do not rely on it
  - @1840 ViewSizeAndInvSize, @1856 BufferSizeAndInvSize
  - @2128 TemporalAAParams = (sample index, sample count (8), jitter x px, jitter y px);
    NDC jitter = (2*x/W, 2*y/H) (same sign for y, matches @416)

## Lessons from v0 (2026-11-01)
- TAA pixel shaders (stable across launches): bytecode FNV-1a 8f74141e5b1d3aa6 (normal) and
  741f2eb17664c147 (responsive). Exactly these two draws match the structural detection in game.
- The main menu starts on a 1920x1080 swap chain and switches to the desktop size when a level loads.
- OptiScaler as a proxy DLL (winmm.dll) breaks the game's rendering (spoofing / NVAPI / device hooks).
  Install it as **nvngx.dll**: the add-on loads it only when FSR is first needed and calls its exported
  NGX functions directly (no nvsdk_ngx_s.lib, which refuses non-NVIDIA adapters).
- The exported NVSDK_NGX_D3D11_Init is the 5-parameter form (with NVSDK_NGX_FeatureCommonInfo).
- UE4's D3D11 state cache: any state the add-on leaves bound is used by later game passes (eye adaptation
  ran with our constants -> black / white image). All add-on work runs inside its own
  ID3DDeviceContextState (SwapDeviceContextState) - this also unbinds the game's DSV so depth is readable.
- Smart App Control (Windows) judges each new unsigned build; a VERSIONINFO resource helped.
- OptiScaler.ini for this game: Dxgi=false, OverrideNvapiDll=false.

## Plan for the upscaler add-on
1. At the first TAA draw: read b1, build camera motion vectors in a compute shader
   (depth -> ClipToTranslatedWorld -> + (PrevPreViewTranslation - PreViewTranslation) ->
   PrevTranslatedWorldToClip), remove both frames' jitter, and use the decoded velocity buffer where it
   is non-zero (UE4 encoding: v = (enc - 32767/65535) / (0.499 * 0.5), verify sign/units).
2. Run FSR 4 / DLSS / XeSS via OptiScaler (reuse styx-upscaler's NGX path) with color = t1, depth = t0,
   jitter from TemporalAAParams, output into the TAA render target; skip both TAA draws.
3. Native AA first. Upscaling via r.ScreenPercentage needs the post chain after TAA at output size: later.
4. The add-on writes the Engine.ini TAA settings itself (like the Styx 1 Engine Fix writes the fps limit).
