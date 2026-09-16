# DLI contre SoA global : l'argument AoSoA, mesuré au lieu d'être affirmé.
#
# La doc affirme depuis toujours que le SoA global « perd la localité quand la population
# grandit, parce que `i-1` est à `nbatch` scalaires ». C'est une hypothèse sur les caches, et
# elle n'avait jamais été vérifiée. Si elle est vraie, le verdict DLI/SoA doit **dépendre de
# la taille du lot**, et les deux courbes doivent se croiser.
#
# Les trois variantes partagent EXACTEMENT le même tableau `(nbatch, nx)` en colonne-majeur,
# donc la même empreinte mémoire totale. Seul l'ordre de parcours change :
#
#   * référence : `b` dehors, `i` dedans  → foulée de `nbatch` sur l'axe de la récurrence ;
#   * SoA       : `i` dehors, `b` dedans  → `b` contigu, `@simd` légal (instances
#                 indépendantes), mais le pas `i` balaie `nbatch` scalaires de chaque tableau ;
#   * DLI       : un paquet traverse toute la récurrence avant le suivant, donc l'ensemble de
#                 travail vaut `nx × P` éléments et non `nbatch`.
#
# La transformation SoA est fidèle : mêmes opérations dans le même ordre par instance, donc le
# résultat doit rester bit-à-bit identique. C'est vérifié avant toute mesure.

using Interleave
using BenchmarkTools
using Printf: @sprintf

include(joinpath(@__DIR__, "..", "test", "kernels.jl"))

const T = Float32

"""Référence : chaque instance de bout en bout, `i` contigu nulle part."""
function thomas_ref!(X, D, U, L, B, S)
    nb, nx = size(X)
    @inbounds for b in 1:nb
        s = D[b, 1]; sm1 = inv(s)
        X[b, 1] = B[b, 1] * sm1
        for i in 2:nx
            S[i] = U[b, i-1] * sm1
            s = D[b, i] - L[b, i] * S[i]
            X[b, i] = B[b, i] - L[b, i] * X[b, i-1]
            sm1 = inv(s)
            X[b, i] *= sm1
        end
        for i in nx-1:-1:1
            X[b, i] -= S[i+1] * X[b, i+1]
        end
    end
    X
end

"""SoA global : `i` dehors, instances dedans. Le `@simd` est légal, les instances étant
indépendantes — c'est la transformation « à la main » que la doc évoque."""
function thomas_soa!(X, D, U, L, B, S, sm1)
    nb, nx = size(X)
    @inbounds begin
        @simd for b in 1:nb
            sm1[b] = inv(D[b, 1])
            X[b, 1] = B[b, 1] * sm1[b]
        end
        for i in 2:nx
            @simd for b in 1:nb
                S[b, i] = U[b, i-1] * sm1[b]
                s = D[b, i] - L[b, i] * S[b, i]
                X[b, i] = B[b, i] - L[b, i] * X[b, i-1]
                sm1[b] = inv(s)
                X[b, i] *= sm1[b]
            end
        end
        for i in nx-1:-1:1
            @simd for b in 1:nb
                X[b, i] -= S[b, i+1] * X[b, i+1]
            end
        end
    end
    X
end

setup_host(nb, nx) = (
    zeros(T, nb, nx),
    T[2 + sinpi(T(b) / 31) / 4 for b in 1:nb, _ in 1:nx],
    T[-1 + cospi(T(b) / 29) / 8 for b in 1:nb, _ in 1:nx],
    T[-1 + sinpi(T(b) / 23) / 8 for b in 1:nb, _ in 1:nx],
    T[sinpi(T(b) / 8) + T(i) / nx for b in 1:nb, i in 1:nx],
)

function one_size(nb, nx, P)
    X, D, U, L, B = setup_host(nb, nx)

    Xr = copy(X); thomas_ref!(Xr, D, U, L, B, zeros(T, nx))
    Xs = copy(X); thomas_soa!(Xs, D, U, L, B, zeros(T, nb, nx), zeros(T, nb))
    Xi = Interleave.Array{T,2,P}(X)
    Di = Interleave.Array{T,2,P}(D); Ui = Interleave.Array{T,2,P}(U)
    Li = Interleave.Array{T,2,P}(L); Bi = Interleave.Array{T,2,P}(B)
    apply!(thomas!, Xi, Di, Ui, Li, Bi; scratch = scratchlike(Xi))

    err_soa = maximum(abs, Xs .- Xr)
    err_dli = maximum(abs(Xi[b, i] - Xr[b, i]) for b in 1:nb, i in 1:nx)

    Sref = zeros(T, nx); Ssoa = zeros(T, nb, nx); sm1 = zeros(T, nb)
    Sdli = scratchlike(Xi)
    # Chaque variante réécrit entièrement sa sortie, donc les échantillons successifs
    # repartent du même problème sans remise à zéro. Budget borné : le plus gros lot alloue
    # plusieurs centaines de Mo et la référence y est lente.
    Xref = copy(X); Xsoa = copy(X)
    t_ref = @belapsed thomas_ref!($Xref, $D, $U, $L, $B, $Sref) samples=5 evals=1 seconds=4
    t_soa = @belapsed thomas_soa!($Xsoa, $D, $U, $L, $B, $Ssoa, $sm1) samples=5 evals=1 seconds=4
    t_dli = @belapsed apply!(thomas!, $Xi, $Di, $Ui, $Li, $Bi; scratch = $Sdli) samples=5 evals=1 seconds=4

    (; nb, t_ref, t_soa, t_dli, err_soa, err_dli)
end

function run(; nx = 64, P = 16,
             batches = (256, 1_024, 4_096, 16_384, 65_536, 262_144))
    println("\n", "="^78)
    println("DLI contre SoA global — Thomas, nx = $nx, P = $P, Float32")
    println("="^78)
    println("Empreinte identique pour les trois variantes ; seul l'ordre de parcours diffère.")
    println()
    println(rpad("nbatch", 10), rpad("octets/i-step", 15), rpad("référence", 12),
            rpad("SoA @simd", 12), rpad("DLI", 12), rpad("DLI/SoA", 10), "accord")
    println("-"^90)
    for nb in batches
        r = one_size(nb, nx, P)
        # Ensemble de travail d'un pas `i` du SoA : ~7 tableaux de nbatch Float32.
        bytes = 7 * nb * sizeof(T)
        ok = (r.err_soa == 0 && r.err_dli == 0) ? "exact" : "soa=$(r.err_soa) dli=$(r.err_dli)"
        println(rpad(nb, 10),
                rpad(bytes < 1024^2 ? @sprintf("%.0f Ko", bytes/1024) : @sprintf("%.1f Mo", bytes/1024^2), 15),
                rpad(@sprintf("%.2f ms", r.t_ref*1e3), 12),
                rpad(@sprintf("%.2f ms", r.t_soa*1e3), 12),
                rpad(@sprintf("%.2f ms", r.t_dli*1e3), 12),
                rpad(@sprintf("%.2f×", r.t_soa/r.t_dli), 10),
                ok)
    end
    println()
    println("DLI/SoA > 1 : le DLI est plus rapide. Si l'hypothèse de cache est juste, ce")
    println("rapport doit CROÎTRE avec nbatch, l'ensemble de travail du DLI restant nx×P.")
    println()
    nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && run()
