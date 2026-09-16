# IIR biquad — récurrence temporelle, vectorisée à travers les canaux audio.
include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(@__DIR__, "..", "test", "kernels.jl"))
const T = Float32
const COEFFS = (0.2f0, 0.4f0, 0.2f0, -0.3f0, 0.1f0)

function reference!(Y, X, c)
    @inbounds for j in axes(Y, 2)
        biquad!(view(Y, :, j), view(X, :, j), c)
    end
    Y
end

function batched(nchan, nsamp, ::Val{P}) where {P}
    Y, X = (Interleave.Array{T}(undef, nchan, nsamp; pack = Val(P)) for _ in 1:2)
    fill!(Y, 0); fill!(X, 1)
    (Y, X)
end

function run(; nchan = 4_096, nsamp = 1_024, rounds = 5)
    Yr, Xr = fill(T(0), nsamp, nchan), fill(T(1), nsamp, nchan)
    Yr0 = copy(Yr)
    sets = map(P -> batched(nchan, nsamp, Val(P)), PACKS)

    variants = Pair{String,Any}["référence" => () -> reference!(Yr, Xr, COEFFS)]
    resets = Function[() -> copyto!(Yr, Yr0)]
    for (P, s) in zip(PACKS, sets)
        push!(variants, "P=$P" => let s = s
            () -> apply!((y, x) -> biquad!(y, x, COEFFS), s...)
        end)
        push!(resets, let s = s, Y0 = deepcopy(s[1])
            () -> copyto!(s[1], Y0)
        end)
        push!(variants, "P=$P threadé" => let s = s
            () -> parallel_apply!((y, x) -> biquad!(y, x, COEFFS), s...;
                            scheduler = StaticScheduler())
        end)
        push!(resets, let s = s, Y0 = deepcopy(s[1])
            () -> copyto!(s[1], Y0)
        end)
    end
    best = interleaved(variants; rounds, resets)

    header("IIR biquad forme directe I (récurrence temporelle)",
           "$nchan canaux × $nsamp échantillons — y[n] dépend de y[n-1] et y[n-2]",
           "échantillons", nchan * nsamp)
    flops = 9 * nchan * nsamp                  # 5 mul + 4 add par échantillon
    extras = Dict("P=$P" => (P == 1 ? "—" :
                  string((E = packtype(T, Val(P)); vectorised(biquad!, Tuple{Vector{E},Vector{E},NTuple{5,T}}, P))))
                  for P in PACKS)
    table(best, ["référence"; ["P=$P" for P in PACKS]], flops, "référence";
          extras, extracol = "LLVM vectorisé")
    println()
    table(best, ["référence"; ["P=$P threadé" for P in PACKS]], flops, "référence")
    best
end
