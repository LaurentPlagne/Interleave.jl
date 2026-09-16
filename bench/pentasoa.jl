# Pentadiagonal : l'écart DLI/SoA s'accroît-il quand le scratch grandit ?
#
# Le tridiagonal donnait un rapport plat à ~2.2×, expliqué par le trafic : la formulation SoA
# doit matérialiser UN workspace `nbatch × nx` en DRAM, celui du DLI faisant `nx × P` et
# restant en L1.
#
# Le pentadiagonal double cette asymétrie : son élimination a besoin de **deux** suites
# auxiliaires (σ et φ) conservées jusqu'à la remontée. Le SoA doit donc matérialiser
# `2 × nbatch × nx`, le DLI `2 × nx × P` — toujours en L1. Si l'explication par le trafic est
# la bonne, le rapport doit monter, et d'une quantité PRÉVISIBLE :
#
#   tri   : DLI ~5 passes, SoA ~9   → 1.80
#   penta : DLI ~7 passes, SoA ~13  → 1.86
#
# soit une hausse réelle mais modeste. C'est cette prédiction chiffrée que le banc teste.
#
# Algorithme : élimination pentadiagonale classique (PTRANS-I).
#   e·x[i-2] + c·x[i-1] + d·x[i] + f·x[i+1] + g·x[i+2] = b[i]

using Interleave
using BenchmarkTools
using Printf: @sprintf

const T = Float32

"""Résolution pentadiagonale d'une instance. `S` est `(n, 2)` : σ en colonne 1, φ en 2."""
function penta!(X, E, C, D, F, G, B, S)
    n = length(X)
    @inbounds begin
        ψ = D[1]
        ψm1 = inv(ψ)
        S[1, 1] = F[1] * ψm1
        S[1, 2] = G[1] * ψm1
        X[1] = B[1] * ψm1

        ρ = C[2]
        ψ = D[2] - ρ * S[1, 1]
        ψm1 = inv(ψ)
        S[2, 1] = (F[2] - ρ * S[1, 2]) * ψm1
        S[2, 2] = G[2] * ψm1
        X[2] = (B[2] - ρ * X[1]) * ψm1

        for i in 3:n
            ρ = C[i] - E[i] * S[i-2, 1]
            ψ = D[i] - E[i] * S[i-2, 2] - ρ * S[i-1, 1]
            ψm1 = inv(ψ)
            S[i, 1] = (F[i] - ρ * S[i-1, 2]) * ψm1
            S[i, 2] = G[i] * ψm1
            X[i] = (B[i] - E[i] * X[i-2] - ρ * X[i-1]) * ψm1
        end

        X[n-1] = X[n-1] - S[n-1, 1] * X[n]
        for i in n-2:-1:1
            X[i] = X[i] - S[i, 1] * X[i+1] - S[i, 2] * X[i+2]
        end
    end
    X
end

"""Référence : une instance de bout en bout sur un tableau batch-major."""
function penta_ref!(X, E, C, D, F, G, B, s1, s2)
    nb, n = size(X)
    @inbounds for b in 1:nb
        ψm1 = inv(D[b, 1])
        s1[1] = F[b, 1] * ψm1; s2[1] = G[b, 1] * ψm1
        X[b, 1] = B[b, 1] * ψm1
        ρ = C[b, 2]
        ψm1 = inv(D[b, 2] - ρ * s1[1])
        s1[2] = (F[b, 2] - ρ * s2[1]) * ψm1; s2[2] = G[b, 2] * ψm1
        X[b, 2] = (B[b, 2] - ρ * X[b, 1]) * ψm1
        for i in 3:n
            ρ = C[b, i] - E[b, i] * s1[i-2]
            ψm1 = inv(D[b, i] - E[b, i] * s2[i-2] - ρ * s1[i-1])
            s1[i] = (F[b, i] - ρ * s2[i-1]) * ψm1; s2[i] = G[b, i] * ψm1
            X[b, i] = (B[b, i] - E[b, i] * X[b, i-2] - ρ * X[b, i-1]) * ψm1
        end
        X[b, n-1] = X[b, n-1] - s1[n-1] * X[b, n]
        for i in n-2:-1:1
            X[b, i] = X[b, i] - s1[i] * X[b, i+1] - s2[i] * X[b, i+2]
        end
    end
    X
end

"""SoA global : `i` dehors, instances dedans. Deux scratch pleins `(nbatch, n)`."""
function penta_soa!(X, E, C, D, F, G, B, S1, S2, ψm1)
    nb, n = size(X)
    @inbounds begin
        @simd for b in 1:nb
            ψm1[b] = inv(D[b, 1])
            S1[b, 1] = F[b, 1] * ψm1[b]; S2[b, 1] = G[b, 1] * ψm1[b]
            X[b, 1] = B[b, 1] * ψm1[b]
        end
        @simd for b in 1:nb
            ρ = C[b, 2]
            ψm1[b] = inv(D[b, 2] - ρ * S1[b, 1])
            S1[b, 2] = (F[b, 2] - ρ * S2[b, 1]) * ψm1[b]
            S2[b, 2] = G[b, 2] * ψm1[b]
            X[b, 2] = (B[b, 2] - ρ * X[b, 1]) * ψm1[b]
        end
        for i in 3:n
            @simd for b in 1:nb
                ρ = C[b, i] - E[b, i] * S1[b, i-2]
                ψm1[b] = inv(D[b, i] - E[b, i] * S2[b, i-2] - ρ * S1[b, i-1])
                S1[b, i] = (F[b, i] - ρ * S2[b, i-1]) * ψm1[b]
                S2[b, i] = G[b, i] * ψm1[b]
                X[b, i] = (B[b, i] - E[b, i] * X[b, i-2] - ρ * X[b, i-1]) * ψm1[b]
            end
        end
        @simd for b in 1:nb
            X[b, n-1] = X[b, n-1] - S1[b, n-1] * X[b, n]
        end
        for i in n-2:-1:1
            @simd for b in 1:nb
                X[b, i] = X[b, i] - S1[b, i] * X[b, i+1] - S2[b, i] * X[b, i+2]
            end
        end
    end
    X
end

"""Produit pentadiagonal, pour le contrôle de résidu."""
function penta_mul!(R, E, C, D, F, G, X)
    nb, n = size(X)
    @inbounds for b in 1:nb, i in 1:n
        acc = D[b, i] * X[b, i]
        i > 2 && (acc += E[b, i] * X[b, i-2])
        i > 1 && (acc += C[b, i] * X[b, i-1])
        i < n && (acc += F[b, i] * X[b, i+1])
        i < n - 1 && (acc += G[b, i] * X[b, i+2])
        R[b, i] = acc
    end
    R
end

# Diagonale strictement dominante : |D| ≥ 8 contre |E|+|C|+|F|+|G| ≤ 4.
function setup(nb, n)
    v(s, a, w) = T[a + w * sinpi(T(b + s) / 31) for b in 1:nb, _ in 1:n]
    (zeros(T, nb, n), v(1, T(-0.5), T(0.2)), v(2, T(-1), T(0.2)), v(3, T(8), T(0.5)),
     v(4, T(-1), T(0.2)), v(5, T(-0.5), T(0.2)),
     T[sinpi(T(b) / 8) + T(i) / n for b in 1:nb, i in 1:n])
end

function one_size(nb, n, P)
    X, E, C, D, F, G, B = setup(nb, n)

    Xr = copy(X); penta_ref!(Xr, E, C, D, F, G, B, zeros(T, n), zeros(T, n))
    R = similar(Xr); penta_mul!(R, E, C, D, F, G, Xr)
    residual = maximum(abs, R .- B)

    Xs = copy(X)
    penta_soa!(Xs, E, C, D, F, G, B, zeros(T, nb, n), zeros(T, nb, n), zeros(T, nb))

    Xi = Interleave.Array{T,2,P}(X)
    Ei, Ci, Di = Interleave.Array{T,2,P}(E), Interleave.Array{T,2,P}(C), Interleave.Array{T,2,P}(D)
    Fi, Gi, Bi = Interleave.Array{T,2,P}(F), Interleave.Array{T,2,P}(G), Interleave.Array{T,2,P}(B)
    # Deux colonnes de scratch par instance : σ et φ. `scratch` accepte tout prototype.
    Sproto = Base.Array{Interleave.packtype(T, Val(P))}(undef, n, 2)
    apply!(penta!, Xi, Ei, Ci, Di, Fi, Gi, Bi; scratch = Sproto)

    err_soa = maximum(abs, Xs .- Xr)
    err_dli = maximum(abs(Xi[b, i] - Xr[b, i]) for b in 1:nb, i in 1:n)

    s1, s2 = zeros(T, n), zeros(T, n)
    S1, S2, ψ = zeros(T, nb, n), zeros(T, nb, n), zeros(T, nb)
    Xref, Xsoa = copy(X), copy(X)
    t_ref = @belapsed penta_ref!($Xref, $E, $C, $D, $F, $G, $B, $s1, $s2) samples=5 evals=1 seconds=4
    t_soa = @belapsed penta_soa!($Xsoa, $E, $C, $D, $F, $G, $B, $S1, $S2, $ψ) samples=5 evals=1 seconds=4
    t_dli = @belapsed apply!(penta!, $Xi, $Ei, $Ci, $Di, $Fi, $Gi, $Bi; scratch = $Sproto) samples=5 evals=1 seconds=4

    (; t_ref, t_soa, t_dli, err_soa, err_dli, residual)
end

function run(; n = 64, P = 16, batches = (16_384, 262_144, 1_048_576, 2_097_152))
    println("\n", "="^86)
    println("Pentadiagonal — DLI contre SoA global, n = $n, P = $P, Float32")
    println("="^86)
    println("Le SoA matérialise DEUX scratch (nbatch × n) ; le DLI en garde deux de (n × P).")
    println("Prédiction du modèle de trafic : ~1.86 contre ~1.80 en tridiagonal.")
    println()
    println(rpad("nbatch", 10), rpad("scratch SoA", 14), rpad("référence", 13),
            rpad("SoA @simd", 13), rpad("DLI", 13), rpad("DLI/SoA", 10), "accord")
    println("-"^96)
    for nb in batches
        r = one_size(nb, n, P)
        ok = (r.err_soa == 0 && r.err_dli == 0) ? @sprintf("exact (résidu %.1e)", r.residual) :
             "soa=$(r.err_soa) dli=$(r.err_dli)"
        by = 2 * nb * n * sizeof(T)
        println(rpad(nb, 10),
                rpad(by < 1024^2 ? @sprintf("%.0f Ko", by/1024) : @sprintf("%.0f Mo", by/1024^2), 14),
                rpad(@sprintf("%.2f ms", r.t_ref*1e3), 13),
                rpad(@sprintf("%.2f ms", r.t_soa*1e3), 13),
                rpad(@sprintf("%.2f ms", r.t_dli*1e3), 13),
                rpad(@sprintf("%.2f×", r.t_soa/r.t_dli), 10),
                ok)
    end
    println()
    nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && run()
