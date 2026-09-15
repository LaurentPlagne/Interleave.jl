# Pipeline vidéo — Sobel spatial 3×3 fusionné à une différence temporelle, instances 2D.
include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(@__DIR__, "..", "test", "kernels.jl"))
const T = Float32
const AB = (0.7f0, 0.3f0)

function reference!(out, curr, prev, ab)
    @inbounds for s in axes(curr, 3)
        sobel_motion!(view(out, :, :, s), view(curr, :, :, s), view(prev, :, :, s), ab)
    end
    out
end

function batched(nstream, H, W, ::Val{P}) where {P}
    O, C, Pr = (Interleave.Array{T}(undef, nstream, H, W; pack = Val(P)) for _ in 1:3)
    fill!(O, 0); fill!(C, 1); fill!(Pr, 0.5)
    (O, C, Pr)
end

function run(; nstream = 256, H = 128, W = 128, rounds = 5)
    outr, currr, prevr = fill(T(0), H, W, nstream), fill(T(1), H, W, nstream), fill(T(0.5), H, W, nstream)
    sets = map(P -> batched(nstream, H, W, Val(P)), PACKS)

    variants = Pair{String,Any}["référence" => () -> reference!(outr, currr, prevr, AB)]
    for (P, s) in zip(PACKS, sets)
        push!(variants, "P=$P" => let s = s
            () -> apply!((o, c, p) -> sobel_motion!(o, c, p, AB), s...)
        end)
        push!(variants, "P=$P threadé" => let s = s
            () -> parallel_apply!((o, c, p) -> sobel_motion!(o, c, p, AB), s...;
                            scheduler = StaticScheduler())
        end)
    end
    best = interleaved(variants; rounds)

    header("Pipeline vidéo : Sobel 3×3 + différence temporelle (instances 2D)",
           "$nstream flux de $(H)×$(W) — stencil spatial fusionné à la détection de mouvement",
           "pixels intérieurs", nstream * (H - 2) * (W - 2))
    flops = 23 * nstream * (H - 2) * (W - 2)   # ≈23 flop/pixel (gradients, magnitude, fusion)
    table(best, ["référence"; ["P=$P" for P in PACKS]], flops, "référence")
    println()
    table(best, ["référence"; ["P=$P threadé" for P in PACKS]], flops, "référence")
    best
end
