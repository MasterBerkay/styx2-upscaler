// RCAS sharpening (the robust contrast-adaptive sharpening of AMD FidelityFX FSR 1) on the tone-mapped image
// right before the game's final pass: a negative lobe on the 4 neighbours, limited per pixel so the result
// never leaves the local min/max range (no halos, no clipping). Same math as FsrRcasF in ffx_fsr1.h, without
// the noise filter.

cbuffer Params : register(b0)
{
    float Sharpness;  // 0..2 (1 = full RCAS, above that stronger with the result kept in the local range)
    float3 Pad;
};

Texture2D<float4> Input : register(t0);
RWTexture2D<float4> Output : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    Input.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
        return;
    const int2 hi = int2(width - 1, height - 1);
    const int2 p = int2(id.xy);
    float4 center = Input.Load(int3(p, 0));
    float3 e = saturate(center.rgb);
    float3 b = saturate(Input.Load(int3(clamp(p + int2(0, -1), 0, hi), 0)).rgb);
    float3 d = saturate(Input.Load(int3(clamp(p + int2(-1, 0), 0, hi), 0)).rgb);
    float3 f = saturate(Input.Load(int3(clamp(p + int2(1, 0), 0, hi), 0)).rgb);
    float3 h = saturate(Input.Load(int3(clamp(p + int2(0, 1), 0, hi), 0)).rgb);

    const float kLimit = 0.25 - 1.0 / 16.0;
    float3 mn4 = min(min(b, d), min(f, h));
    float3 mx4 = max(max(b, d), max(f, h));
    float3 hitMin = mn4 / (4.0 * mx4 + 1e-5);
    float3 hitMax = (1.0 - mx4) / (4.0 * mn4 - 4.0 - 1e-5);
    float3 lobeRgb = max(-hitMin, hitMax);
    float lobe = max(-kLimit, min(max(lobeRgb.r, max(lobeRgb.g, lobeRgb.b)), 0.0)) * min(Sharpness, 1.0);
    float3 color = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);
    // Above 1 (RCAS is at its own limit): push the RCAS result further away from the neighbour average and allow
    // a controlled overshoot of up to half the local contrast at 2 - inside the plain local range nothing would
    // change at peaks and dips, which is where sharpening shows.
    if (Sharpness > 1.0)
    {
        const float extra = Sharpness - 1.0;
        const float3 lo = min(mn4, e), hi = max(mx4, e);
        const float3 overshoot = (hi - lo) * 0.5 * extra;
        color = clamp(color + (color - (b + d + f + h) * 0.25) * extra * 2.0, lo - overshoot, hi + overshoot);
        color = saturate(color);
    }
    Output[id.xy] = float4(color, center.a);
}
