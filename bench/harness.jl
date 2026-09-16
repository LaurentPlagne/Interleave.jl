# Harnais commun aux bancs.
#
# Méthode (julia-recommandations.md §3.3) : A/B **entrelacé** — toutes les variantes sont
# jouées à chaque tour, on retient le minimum par variante. Cela annule les dérives lentes
# de charge machine, contrairement à « N fois la variante A puis N fois la variante B ».
# La charge est relevée avant la mesure, et aucune conclusion n'est tirée d'un chiffre seul
# sans corroboration structurelle (le `<P x float>` dans le LLVM émis).

using Interleave
using BenchmarkTools
using InteractiveUtils: code_llvm
using Printf: @sprintf

"""Tampon préalloué partagé — type nommé appelable, pas une closure (qui serait boxée).
Valide en séquentiel seulement : en threadé, chaque tâche doit avoir le sien."""
struct SharedBuf{S}
    buf::S
end
(s::SharedBuf)() = s.buf

"""Le noyau `f` émet-il bien des instructions vectorielles de largeur `P` ?"""
function vectorised(f, argtypes, P, ::Type{T} = Float32) where {T}
    io = IOBuffer()
    code_llvm(io, f, argtypes; debuginfo = :none)
    occursin("<$P x $(T === Float32 ? "float" : "double")>", String(take!(io)))
end

"""Construit un essai BenchmarkTools spécialisé sur le type de l'appelable."""
benchmarkable(f) = BenchmarkTools.@benchmarkable $f() samples=1 evals=1

@inline _reset_benchmark!(::Nothing, ::Int) = nothing
@inline _reset_benchmark!(resets, i::Int) = (resets[i](); nothing)

"""Joue toutes les variantes à chaque tour ; rend le meilleur temps de chacune.

BenchmarkTools fournit l'isolation des globales et la mesure en nanosecondes. Nous gardons
néanmoins l'ordre A/B entrelacé : un `Trial` d'un échantillon est collecté pour chaque
variante à chaque tour, au lieu d'épuiser une variante avant de commencer la suivante.
"""
function interleaved(variants; rounds = 8, resets = nothing)
    resets === nothing || length(resets) == length(variants) ||
        throw(ArgumentError("one reset callback is required per benchmark variant"))
    for (i, (_, f)) in enumerate(variants)
        _reset_benchmark!(resets, i)
        f()                                  # échauffement : compilation + pages
    end
    trials = map(variants) do pair
        name, f = pair
        name => benchmarkable(f)
    end
    best = Dict{String,Float64}()
    for _ in 1:rounds, (i, (name, trial)) in enumerate(trials)
        # Every trial has one evaluation, so reset work stays outside the timed
        # region while each sample starts from the same state. This is essential
        # for in-place recurrences such as Thomas and Black-Scholes.
        _reset_benchmark!(resets, i)
        t = BenchmarkTools.minimum(BenchmarkTools.run(trial)).time / 1e9
        best[name] = min(get(best, name, Inf), t)
    end
    best
end

function header(title, detail, work_label, work)
    println("\n", "="^70)
    println(title)
    println("="^70)
    println(detail)
    println(work_label, " = ", work, " | threads = ", Threads.nthreads(),
            " | ", strip(split(read(`uptime`, String), "load averages:")[end]))
    println()
end

"""Tableau : temps, débit, accélération contre `baseline`, et colonne libre."""
function table(best, order, flops, baseline; extras = Dict{String,String}(), extracol = "")
    t0 = best[baseline]
    println(rpad("variante", 18), rpad("temps", 11), rpad("GFlop/s", 10),
            rpad("accél.", 9), extracol)
    println("-"^(48 + length(extracol)))
    for name in order
        t = best[name]
        println(rpad(name, 18),
                rpad(@sprintf("%.2f ms", t * 1e3), 11),
                rpad(@sprintf("%.1f", flops / t / 1e9), 10),
                rpad(@sprintf("%.2f×", t0 / t), 9),
                get(extras, name, ""))
    end
end

const PACKS = (1, 2, 4, 8, 16, 32)
