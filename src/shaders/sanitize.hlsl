// Copies the upscaler's output, replacing NaN / Inf, before it goes back into the game's post-processing.

Texture2D<float4> Input : register(t0);
RWTexture2D<float4> Output : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    float4 c = Input.Load(int3(id.xy, 0));
    if (any(isnan(c)) || any(isinf(c)))
        c = float4(0, 0, 0, 1);
    Output[id.xy] = float4(clamp(c.rgb, 0.0, 64000.0), saturate(c.a));
}
