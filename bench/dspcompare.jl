# Interleave contre DSP.jl sur un banc de filtres biquad.
#
# La doc affirmait que DSP.jl est « une vraie réponse, bien optimisée » sans jamais le
# mesurer. Ce banc répare ça. La comparaison doit être honnête sur trois points :
#
#   * mêmes coefficients, même `Float32`, même travail total ;
#   * chaque bibliothèque dans SA disposition naturelle — DSP filtre les colonnes d'une
#     matrice `(ns, nb)`, Interleave travaille en batch-major `(nb, ns)`. Transposer l'une
#     pour l'autre mesurerait la transposition, pas le filtrage ;
#   * accord numérique vérifié avant toute mesure de temps.
#
# `DSP.filt(b, a, X)` sur une matrice filtre **chaque colonne indépendamment** : c'est
# exactement un banc de filtres, en un seul appel. C'est donc bien le concurrent direct.

using Interleave
using DSP
using BenchmarkTools
using Printf: @sprintf

include(joinpath(@__DIR__, "..", "test", "kernels.jl"))

const T = Float32
const COEFFS = (T(0.2), T(0.4), T(0.2), T(-0.3), T(0.1))   # b0 b1 b2 a1 a2
const BC = T[COEFFS[1], COEFFS[2], COEFFS[3]]
const AC = T[one(T), COEFFS[4], COEFFS[5]]

biquad_bank!(Y, X) = biquad!(Y, X, COEFFS)

"""Référence naïve : une boucle Julia ordinaire sur les canaux, sans annotation."""
function naive_bank!(Y, X)
    nb, ns = size(X)
    @inbounds for c in 1:nb
        b0, b1, b2, a1, a2 = COEFFS
        x1 = zero(T); x2 = zero(T); y1 = zero(T); y2 = zero(T)
        for n in 1:ns
            x0 = X[c, n]
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            Y[c, n] = y0
            x2 = x1; x1 = x0
            y2 = y1; y1 = y0
        end
    end
    Y
end

function run(; nb = 1_024, ns = 4_096, packs = (1, 4, 8, 16, 32))
    println("\n", "="^72)
    println("Banc de biquads : $nb canaux indépendants × $ns échantillons, Float32")
    println("="^72)

    # Données batch-major (nb, ns) et leur transposée (ns, nb) pour DSP.
    Xb = T[sinpi(T(c) / 17) + T(n) / ns for c in 1:nb, n in 1:ns]
    Xd = permutedims(Xb)                        # (ns, nb), canaux en colonnes
    Yb = zeros(T, nb, ns)

    # --- accord numérique, avant toute mesure -------------------------------------
    naive_bank!(Yb, Xb)
    Yd = DSP.filt(BC, AC, Xd)
    dsp_vs_naive = maximum(abs, permutedims(Yd) .- Yb)

    Xi = Interleave.Array{T,2,16}(Xb)
    Yi = similar(Xi)
    fill!(Yi, 0)
    apply!(biquad_bank!, Yi, Xi)
    dli_vs_naive = maximum(abs, [Yi[c, n] - Yb[c, n] for c in 1:nb, n in 1:ns])

    println("accord   DSP vs naïf : ", dsp_vs_naive)
    println("accord   DLI vs naïf : ", dli_vs_naive, "  (doit être exactement 0)")
    println()

    # --- temps ---------------------------------------------------------------------
    t_naive = @belapsed naive_bank!($Yb, $Xb)
    t_dsp   = @belapsed DSP.filt($BC, $AC, $Xd)
    # `filt!` évite l'allocation de sortie, pour ne pas facturer à DSP un malloc que la
    # version Interleave ne paie pas.
    Ydout = similar(Xd)
    t_dspi  = @belapsed DSP.filt!($Ydout, $BC, $AC, $Xd)

    results = ["boucle naïve" => t_naive, "DSP.filt" => t_dsp, "DSP.filt! (en place)" => t_dspi]

    for P in packs
        Xp = Interleave.Array{T,2,P}(Xb)
        Yp = similar(Xp); fill!(Yp, 0)
        t = @belapsed apply!(biquad_bank!, $Yp, $Xp)
        push!(results, "Interleave P=$P" => t)
    end

    t0 = t_naive
    println(rpad("variante", 24), rpad("temps", 13), rpad("vs naïf", 11), "vs DSP.filt!")
    println("-"^62)
    for (name, t) in results
        println(rpad(name, 24),
                rpad(@sprintf("%.3f ms", t * 1e3), 13),
                rpad(@sprintf("%.2f×", t0 / t), 11),
                @sprintf("%.2f×", t_dspi / t))
    end
    println()
    nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && run()
