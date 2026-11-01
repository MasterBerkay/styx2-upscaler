// Motion vectors and depth for the upscaler, from UE 4.13's TAA inputs.
//
// UE4 renders in "translated world" space (world + PreViewTranslation) with row-vector math and reversed Z.
// Static pixels: rebuild the translated world position from depth with this frame's (jittered)
// ClipToTranslatedWorld, move it into last frame's translated space and project it with last frame's
// (jittered) TranslatedWorldToClip. Moving objects: UE4's velocity buffer (NDC delta current - previous).
// Both are made jitter-free. Output: motion in UV units pointing from the current pixel to its previous
// position (the FSR / DLSS convention) and depth as R32_FLOAT.

cbuffer Params : register(b0)
{
    row_major float4x4 ClipToTranslatedWorld;      // this frame, jittered
    row_major float4x4 PrevTranslatedWorldToClip;  // previous frame, jittered
    float4 TranslationDelta;   // xyz = PrevPreViewTranslation - PreViewTranslation
    float2 RenderSize;
    float2 JitterNdc;          // this frame's projection jitter (NDC)
    float2 PrevJitterNdc;      // previous frame's projection jitter (NDC)
    float DebugMode;           // 0 off, 1 motion vector colors
    float Pad;
};

Texture2D<float> SceneDepth : register(t0);
Texture2D<float2> Velocity : register(t1);    // UE4 encoded velocity, (0,0) = none written
Texture2D<float4> SceneColor : register(t2);  // only for the debug view
RWTexture2D<float2> MotionOut : register(u0);
RWTexture2D<float> DepthOut : register(u1);
RWTexture2D<float4> DebugOut : register(u2);
RWTexture2D<float4> ColorOut : register(u3);  // scene color without NaN / Inf, the upscaler's input

// UE4's TAA quietly drops NaN / Inf pixels; FSR / DLSS would spread them over the image and into the
// game's eye adaptation. Replace them, and keep values in half-float range.
float4 Sanitize(float4 c)
{
    if (any(isnan(c)) || any(isinf(c)))
        return float4(0, 0, 0, 1);
    return float4(clamp(c.rgb, 0.0, 64000.0), saturate(c.a));
}

static const float kVelocityInvDiv = 1.0 / (0.499 * 0.5);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)RenderSize.x || id.y >= (uint)RenderSize.y)
        return;

    float2 uv = (float2(id.xy) + 0.5) / RenderSize;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);  // as rendered (jittered projection)
    float depth = SceneDepth.Load(int3(id.xy, 0));
    DepthOut[id.xy] = depth;
    ColorOut[id.xy] = Sanitize(SceneColor.Load(int3(id.xy, 0)));

    // Camera motion: NDC of this surface point last frame, both positions without jitter.
    float4 h = mul(float4(ndc, depth, 1.0), ClipToTranslatedWorld);
    float4 prevClip = abs(h.w) < 1e-7 ? mul(float4(h.xyz, 0.0), PrevTranslatedWorldToClip)  // sky: direction only
                                      : mul(float4(h.xyz / h.w + TranslationDelta.xyz, 1.0), PrevTranslatedWorldToClip);
    float2 prevNdc = prevClip.xy / prevClip.w - PrevJitterNdc;
    float2 curNdc = ndc - JitterNdc;
    float2 deltaNdc = curNdc - prevNdc;  // current - previous

    // Moving objects write their own velocity (current - previous NDC, including the jitter difference).
    float2 enc = Velocity.Load(int3(id.xy, 0));
    const float2 cameraDeltaNdc = deltaNdc;
    if (enc.x > 0.0)
        deltaNdc = (enc * kVelocityInvDiv - (32767.0 / 65535.0) * kVelocityInvDiv) - (JitterNdc - PrevJitterNdc);

    if (DebugMode > 2.5)
    {
        // Velocity check: where an object wrote velocity, compare it with the camera-only motion.
        // Green = within 1 px (static object), yellow = within 4 px, red = more; grey = no object velocity.
        float3 c = saturate(dot(SceneColor.Load(int3(id.xy, 0)).rgb, float3(0.299, 0.587, 0.114))).xxx * 0.4;
        if (enc.x > 0.0)
        {
            float px = length((deltaNdc - cameraDeltaNdc) * 0.5 * RenderSize);
            c = px < 1.0 ? float3(0.0, 0.8, 0.0) : (px < 4.0 ? float3(0.9, 0.8, 0.0) : float3(0.9, 0.0, 0.0));
        }
        DebugOut[id.xy] = float4(c, 1.0);
    }

    // NDC delta -> UV offset towards the previous position (UV y points down).
    float2 motion = float2(-deltaNdc.x * 0.5, deltaNdc.y * 0.5);
    MotionOut[id.xy] = motion;

    if (DebugMode > 0.5 && DebugMode < 1.5)
    {
        float2 px = motion * RenderSize / 16.0;  // 16 px of motion = full color
        float3 viz = float3(saturate(0.5 + px.x), saturate(0.5 + px.y), enc.x > 0.0 ? 1.0 : 0.5);
        float luma = dot(SceneColor.Load(int3(id.xy, 0)).rgb, float3(0.299, 0.587, 0.114));
        DebugOut[id.xy] = float4(lerp(viz, saturate(luma).xxx, 0.3), 1.0);
    }
}
