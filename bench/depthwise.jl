# Convolution depthwise 3×3 — instances 2D (cartes de caractéristiques), pas de récurrence.
# Cas de contrôle intéressant : ici le compilateur PEUT vectoriser la référence le long
# de la dimension spatiale. Le gain DLI y est donc structurellement plus faible.
include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(@__DIR__, "..", "test", "kernels.jl"))
const T = Float32
const W3 = T.((1, 2, 1, 2, 4, 2, 1, 2, 1) ./ 16)

function reference!(out, inp, w)
    @inbounds for c in axes(inp, 3)
        depthwise3x3!(view(out, :, :, c), view(inp, :, :, c), w)
    end
    out
end

function batched(nchan, H, W, ::Val{P}) where {P}
    O, I = (Interleave.Array{T}(undef, nchan, H, W; pack = Val(P)) for _ in 1:2)
    fill!(O, 0); fill!(I, 1)
    (O, I)
end

function run(; nchan = 512, H = 64, W = 64, rounds = 5)
    outr, inpr = fill(T(0), H, W, nchan), fill(T(1), H, W, nchan)
    outr0 = copy(outr)
    sets = map(P -> batched(nchan, H, W, Val(P)), PACKS)

    variants = Pair{String,Any}["référence" => () -> reference!(outr, inpr, W3)]
    resets = Function[() -> copyto!(outr, outr0)]
    for (P, s) in zip(PACKS, sets)
        push!(variants, "P=$P" => let s = s
            () -> apply!((o, i) -> depthwise3x3!(o, i, W3), s...)
        end)
        push!(resets, let s = s, O0 = deepcopy(s[1])
            () -> copyto!(s[1], O0)
        end)
        push!(variants, "P=$P threadé" => let s = s
            () -> parallel_apply!((o, i) -> depthwise3x3!(o, i, W3), s...;
                            scheduler = StaticScheduler())
        end)
        push!(resets, let s = s, O0 = deepcopy(s[1])
            () -> copyto!(s[1], O0)
        end)
    end
    best = interleaved(variants; rounds, resets)

    header("Convolution depthwise 3×3 (stencil 2D, sans récurrence)",
           "$nchan canaux de $(H)×$(W) — instances 2D, le compilateur sait déjà vectoriser la référence",
           "pixels intérieurs", nchan * (H - 2) * (W - 2))
    flops = 17 * nchan * (H - 2) * (W - 2)     # 9 mul + 8 add par pixel
    # Type de vue obtenu par `typeof` sur une instance réelle : le reconstruire à la
    # main est une source d'erreur (paramètre de contiguïté notamment).
    viewtype(s) = typeof(packet(s[1], 1))
    extras = Dict("P=$P" => (P == 1 ? "—" :
                  string(vectorised(depthwise3x3!, Tuple{viewtype(s),viewtype(s),NTuple{9,T}}, P)))
                  for (P, s) in zip(PACKS, sets))
    table(best, ["référence"; ["P=$P" for P in PACKS]], flops, "référence";
          extras, extracol = "LLVM vectorisé")
    println()
    table(best, ["référence"; ["P=$P threadé" for P in PACKS]], flops, "référence")
    best
end
