# Black-Scholes par différences finies, Crank-Nicolson — double récurrence :
# une boucle en temps, et dans chaque pas de temps un balayage de Thomas.
include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(@__DIR__, "..", "test", "kernels.jl"))
const T = Float32

function reference!(V, D, U, L, RHS, S, nt)
    @inbounds for j in axes(V, 2)
        blackscholes_cn!(view(V, :, j), view(D, :, j), view(U, :, j),
                         view(L, :, j), view(RHS, :, j), S, nt)
    end
    V
end

function batched(nopt, ngrid, ::Val{P}) where {P}
    V, D, U, L, RHS = (Interleave.Array{T}(undef, nopt, ngrid; pack = Val(P)) for _ in 1:5)
    fill!(V, 1); fill!(D, 2.05); fill!(U, -0.5); fill!(L, -0.5); fill!(RHS, 0)
    (V, D, U, L, RHS)
end

function run(; nopt = 16_384, ngrid = 64, nt = 32, rounds = 5)
    mkm(v) = fill(T(v), ngrid, nopt)
    Vr, Dr, Ur, Lr, Rr = mkm(1), mkm(2.05), mkm(-0.5), mkm(-0.5), mkm(0)
    Sr = Vector{T}(undef, ngrid)
    sets = map(P -> batched(nopt, ngrid, Val(P)), PACKS)
    bufs = map(P -> (E = packtype(T, Val(P)); SharedBuf(fill!(Array{E}(undef, ngrid), zero(E)))), PACKS)

    variants = Pair{String,Any}["référence" => () -> reference!(Vr, Dr, Ur, Lr, Rr, Sr, nt)]
    for (P, s, b) in zip(PACKS, sets, bufs)
        push!(variants, "P=$P" => let s = s, b = b
            () -> apply!((v, d, u, l, r, sc) -> blackscholes_cn!(v, d, u, l, r, sc, nt),
                            s...; scratch = b)
        end)
        push!(variants, "P=$P threadé" => let s = s
            sc = similar(instance(s[1], 1))
            () -> parallel_apply!((v, d, u, l, r, w) -> blackscholes_cn!(v, d, u, l, r, w, nt),
                            s...; scratch = sc, scheduler = StaticScheduler())
        end)
    end
    best = interleaved(variants; rounds)

    header("Black-Scholes Crank-Nicolson (double récurrence)",
           "$nopt options × grille $ngrid × $nt pas de temps — Thomas imbriqué dans la boucle temporelle",
           "points-pas", nopt * ngrid * nt)
    flops = 15 * nopt * ngrid * nt             # ≈6 (explicite) + 9 (Thomas) par point et par pas
    table(best, ["référence"; ["P=$P" for P in PACKS]], flops, "référence")
    println()
    table(best, ["référence"; ["P=$P threadé" for P in PACKS]], flops, "référence")
    best
end
