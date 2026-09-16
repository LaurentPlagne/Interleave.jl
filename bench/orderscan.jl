# « Plus la récurrence a de prédécesseurs, plus le DLI gagne » — balayage de l'ordre.
#
# Le tridiagonal (≈7 flux batch-majeurs) donnait 2.2×, le pentadiagonal (≈11–13 flux) 11.8×.
# L'hypothèse est que le nombre de PRÉDÉCESSEURS gouverne l'écart. Un IIR d'ordre `M` en fait
# un paramètre continu :
#
#     y[n] = b₀·x[n] + Σₖ bₖ·x[n−k] − Σₖ aₖ·y[n−k]        k = 1..M
#
# * DLI : l'état (2M valeurs) tient dans des REGISTRES, l'instance est contiguë. L'ensemble
#   de travail vaut `ns × P` par tableau, indépendamment de `M` et de `nbatch`.
# * SoA : `y[n−k]` et `x[n−k]` sont `Y[b, n−k]` et `X[b, n−k]`, donc **2M+1 flux
#   batch-majeurs simultanés**, chacun avançant de `nbatch × 4` octets entre deux `n`.
#
# Prédiction : l'écart DLI/SoA doit croître avec `M`, et décrocher quand 2M+1 dépasse ce que
# les prefetchers soutiennent. C'est exactement ce que le pentadiagonal a montré ponctuellement.

using Interleave
using BenchmarkTools
using Printf: @sprintf

const T = Float32

# Expression déroulée à la compilation : même ordre d'opérations pour le scalaire, le DLI et
# le SoA, donc bit-exactitude exigible entre les trois (AGENTS.md, invariant 1).
@generated function _step(b::NTuple{Mp1,S}, a::NTuple{M,S}, x0,
                          xs::NTuple{M,E}, ys::NTuple{M,E}) where {Mp1,M,S,E}
    ex = :(b[1] * x0)
    for k in 1:M
        ex = :($ex + b[$k + 1] * xs[$k] - a[$k] * ys[$k])
    end
    ex
end

"""IIR d'ordre `M`, une instance. L'état est un `NTuple` — donc en registres."""
function iir!(Y, X, b::NTuple{Mp1,S}, a::NTuple{M,S}) where {Mp1,M,S}
    E = eltype(Y)
    xs = ntuple(_ -> zero(E), Val(M))
    ys = ntuple(_ -> zero(E), Val(M))
    @inbounds for n in eachindex(Y)
        x0 = X[n]
        y0 = _step(b, a, x0, xs, ys)
        Y[n] = y0
        xs = (x0, Base.front(xs)...)
        ys = (y0, Base.front(ys)...)
    end
    Y
end

"""Référence : une instance de bout en bout sur un tableau batch-major."""
function iir_ref!(Y, X, b::NTuple{Mp1,S}, a::NTuple{M,S}) where {Mp1,M,S}
    nb, ns = size(Y)
    @inbounds for bi in 1:nb
        xs = ntuple(_ -> zero(T), Val(M))
        ys = ntuple(_ -> zero(T), Val(M))
        for n in 1:ns
            x0 = X[bi, n]
            y0 = _step(b, a, x0, xs, ys)
            Y[bi, n] = y0
            xs = (x0, Base.front(xs)...)
            ys = (y0, Base.front(ys)...)
        end
    end
    Y
end

# Régime établi : `n > M`, tout l'historique est en mémoire, la boucle instance vectorise.
@generated function _soa_step(b::NTuple{Mp1,S}, a::NTuple{M,S}, X, Y, bi, n) where {Mp1,M,S}
    ex = :(b[1] * X[bi, n])
    for k in 1:M
        ex = :($ex + b[$k + 1] * X[bi, n - $k] - a[$k] * Y[bi, n - $k])
    end
    ex
end

"""SoA global : `n` dehors, instances dedans. L'historique est relu depuis `X` et `Y`,
donc 2M+1 flux batch-majeurs concurrents."""
function iir_soa!(Y, X, b::NTuple{Mp1,S}, a::NTuple{M,S}) where {Mp1,M,S}
    nb, ns = size(Y)
    @inbounds begin
        # Amorçage : les M premiers échantillons ont un historique incomplet. O(M·nb) contre
        # O(ns·nb) au total, soit moins de 0.5 % du travail pour ns = 4096, M ≤ 16.
        for n in 1:min(M, ns)
            for bi in 1:nb
                acc = b[1] * X[bi, n]
                for k in 1:M
                    # Hors plage = historique nul, et surtout MÊME associativité que
                    # `_step` : `(acc + b·x) - a·y`, et non `acc + (b·x - a·y)`. Les deux
                    # diffèrent d'un ulp en Float32, ce qui suffit à casser l'égalité exacte.
                    inr = n - k >= 1
                    xv = inr ? X[bi, n-k] : zero(T)
                    yv = inr ? Y[bi, n-k] : zero(T)
                    acc = acc + b[k+1] * xv
                    acc = acc - a[k] * yv
                end
                Y[bi, n] = acc
            end
        end
        for n in M+1:ns
            @simd for bi in 1:nb
                Y[bi, n] = _soa_step(b, a, X, Y, bi, n)
            end
        end
    end
    Y
end

"""Coefficients stables pour l'ordre `M` : pôles très amortis, donc pas de divergence."""
function coeffs(M)
    b = ntuple(k -> T(0.4) / T(k), Val(M + 1))
    a = ntuple(k -> T(-0.5)^k / T(2k), Val(M))
    b, a
end

function one_order(M, nb, ns, P)
    b, a = coeffs(M)
    X = T[sinpi(T(c) / 17) + T(n) / ns for c in 1:nb, n in 1:ns]

    Yr = zeros(T, nb, ns); iir_ref!(Yr, X, b, a)
    Ys = zeros(T, nb, ns); iir_soa!(Ys, X, b, a)
    Xi = Interleave.Array{T,2,P}(X)
    Yi = similar(Xi); fill!(Yi, 0)
    apply!((y, x) -> iir!(y, x, b, a), Yi, Xi)

    err_soa = maximum(abs, Ys .- Yr)
    err_dli = maximum(abs(Yi[c, n] - Yr[c, n]) for c in 1:nb, n in 1:ns)
    finite = all(isfinite, Yr)

    Yref = zeros(T, nb, ns); Ysoa = zeros(T, nb, ns)
    t_ref = @belapsed iir_ref!($Yref, $X, $b, $a) samples=5 evals=1 seconds=4
    t_soa = @belapsed iir_soa!($Ysoa, $X, $b, $a) samples=5 evals=1 seconds=4
    t_dli = @belapsed apply!((y, x) -> iir!(y, x, $b, $a), $Yi, $Xi) samples=5 evals=1 seconds=4

    (; t_ref, t_soa, t_dli, err_soa, err_dli, finite)
end

function run(; nb = 65_536, ns = 1_024, P = 16, orders = (1, 2, 4, 8, 16))
    println("\n", "="^94)
    println("IIR d'ordre M — DLI contre SoA global.  $nb canaux × $ns échantillons, P = $P")
    println("="^94)
    println("Le SoA relit l'historique en mémoire : 2M+1 flux batch-majeurs concurrents.")
    println("Le DLI garde 2M valeurs d'état en registres, quel que soit M.")
    println()
    println(rpad("M", 4), rpad("flux SoA", 10), rpad("référence", 13), rpad("SoA @simd", 13),
            rpad("DLI", 13), rpad("DLI/SoA", 10), "accord")
    println("-"^94)
    for M in orders
        r = one_order(M, nb, ns, P)
        ok = !r.finite ? "DIVERGE" :
             (r.err_soa == 0 && r.err_dli == 0) ? "exact" :
             @sprintf("soa=%.2e dli=%.2e", r.err_soa, r.err_dli)
        println(rpad(M, 4), rpad(2M + 1, 10),
                rpad(@sprintf("%.2f ms", r.t_ref*1e3), 13),
                rpad(@sprintf("%.2f ms", r.t_soa*1e3), 13),
                rpad(@sprintf("%.2f ms", r.t_dli*1e3), 13),
                rpad(@sprintf("%.2f×", r.t_soa/r.t_dli), 10), ok)
    end
    println()
    nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && run()
