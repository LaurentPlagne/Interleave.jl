# Thomas tridiagonal — récurrence pure sur un lot de systèmes indépendants.
include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(@__DIR__, "..", "test", "kernels.jl"))
const T = Float32

"Référence : ce qu'un utilisateur Julia écrit naturellement — une colonne par système."
function reference!(X, D, U, L, B, S)
    @inbounds for j in axes(X, 2)
        thomas!(view(X, :, j), view(D, :, j), view(U, :, j),
                view(L, :, j), view(B, :, j), S)
    end
    X
end

"""Alternative SoA globale : on réécrit l'algorithme pour vectoriser sur le lot.

La dimension des problèmes est contiguë, mais deux étapes de récurrence sont séparées par
`nsys` scalaires. Ce cas mesure précisément le compromis que DLI/AoSoA cherche à éviter.
"""
function soa_reference!(X, D, U, L, B, S)
    jobs = axes(X, 1)
    n = size(X, 2)
    @inbounds begin
        @simd for job in jobs
            invpivot = inv(D[job, 1])
            X[job, 1] = B[job, 1] * invpivot
            S[job, 1] = invpivot
        end
        for i in 2:n
            @simd for job in jobs
                factor = U[job, i - 1] * S[job, i - 1]
                S[job, i] = inv(D[job, i] - L[job, i] * factor)
                rhs = B[job, i] - L[job, i] * X[job, i - 1]
                X[job, i] = rhs * S[job, i]
            end
        end
        for i in n-1:-1:1
            @simd for job in jobs
                factor = U[job, i] * S[job, i]
                X[job, i] -= factor * X[job, i + 1]
            end
        end
    end
    X
end

function batched(nsys, nx, ::Val{P}) where {P}
    X, D, U, L, B = (Interleave.Array{T}(undef, nsys, nx; pack = Val(P)) for _ in 1:5)
    fill!(X, 0); fill!(D, 2); fill!(U, -1); fill!(L, -1); fill!(B, 1)
    (X, D, U, L, B)
end

function run(; nsys = 65_536, nx = 64, rounds = 5)
    mkm(v) = fill(T(v), nx, nsys)
    Xr, Dr, Ur, Lr, Br = mkm(0), mkm(2), mkm(-1), mkm(-1), mkm(1)
    Sr = Vector{T}(undef, nx)
    mksoa(v) = fill(T(v), nsys, nx)
    Xsoa, Dsoa, Usoa, Lsoa, Bsoa, Ssoa =
        mksoa(0), mksoa(2), mksoa(-1), mksoa(-1), mksoa(1), mksoa(0)
    sets = map(P -> batched(nsys, nx, Val(P)), PACKS)
    bufs = map(P -> (E = packtype(T, Val(P)); SharedBuf(fill!(Array{E}(undef, nx), zero(E)))), PACKS)

    variants = Pair{String,Any}["référence" => () -> reference!(Xr, Dr, Ur, Lr, Br, Sr)]
    push!(variants, "SoA global" =>
          () -> soa_reference!(Xsoa, Dsoa, Usoa, Lsoa, Bsoa, Ssoa))
    for (P, s, b) in zip(PACKS, sets, bufs)
        push!(variants, "P=$P" => let s = s, b = b
            () -> apply!(thomas!, s...; scratch = b)
        end)
        push!(variants, "P=$P threadé" => let s = s, P = P
            sc = similar(instance(s[1], 1))
            () -> parallel_apply!(thomas!, s...; scratch = sc, scheduler = StaticScheduler())
        end)
    end
    best = interleaved(variants; rounds)

    header("Thomas tridiagonal (récurrence)",
           "$nsys systèmes de taille $nx — récurrence stricte, aucun compilateur ne la vectorise",
           "inconnues", nsys * nx)
    flops = 8 * nsys * nx                      # ≈8 flop/inconnue (aller + retour)
    extras = Dict("P=$P" => (P == 1 ? "—" : string(vectorised(thomas!, NTuple{6,Vector{packtype(T, Val(P))}}, P)))
                  for P in PACKS)
    table(best, ["référence", "SoA global", ("P=$P" for P in PACKS)...], flops, "référence";
          extras, extracol = "LLVM vectorisé")
    println()
    table(best, ["référence"; ["P=$P threadé" for P in PACKS]], flops, "référence")
    best
end
