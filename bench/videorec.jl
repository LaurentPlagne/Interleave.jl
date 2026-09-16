# Vidéo : ce qui se passe quand on ajoute une récurrence à un noyau qui perdait.
#
# Le pipeline Sobel + mouvement est un STENCIL pur : LLVM le vectorise déjà (43.4 GFlop/s),
# et le DLI y perd (0.88×). C'est le cas négatif documenté.
#
# On lui ajoute ce que fait tout vrai pipeline d'image : un lissage **récursif** le long des
# lignes, de la famille Deriche / van Vliet — un IIR d'ordre `M` appliqué ligne par ligne,
# puis le Sobel sur le résultat. `M = 0` redonne exactement le noyau d'origine.
#
# L'instance est ici une image 2-D, pas un signal : on vérifie donc que l'effet vu sur
# l'audio ne tient pas à la dimension 1.
#
# Prédiction : le verdict doit BASCULER de la perte au gain dès que `M` croît, la récurrence
# le long des lignes étant invectorisable par le compilateur.

using Interleave
using BenchmarkTools
using Printf: @sprintf

include(joinpath(@__DIR__, "..", "test", "kernels.jl"))

const T = Float32
const AB = (T(0.7), T(0.3))

# `sum` refuse un tuple vide ; M = 0 doit rendre le noyau d'origine, de gain unité.
@inline _csum(::Tuple{}) = zero(T)
@inline _csum(c::Tuple) = sum(c)

# Idem pour le décalage d'historique : `Base.front` refuse le tuple vide.
@inline _shift(_, ::Tuple{}) = ()
@inline _shift(y0, ys::Tuple) = (y0, Base.front(ys)...)

# `NTuple{M,Any}` et non `NTuple{M,S}` : pour M = 0 le tuple est `Tuple{}` et un paramètre
# de type d'élément ne se lierait pas, donc la méthode ne serait pas sélectionnée.
@generated function _rstep(c::NTuple{M,Any}, acc0, ys::NTuple{M,Any}) where {M}
    ex = :(acc0)
    for k in 1:M
        ex = :($ex - c[$k] * ys[$k])
    end
    ex
end

"""Lissage récursif d'ordre `M` le long des lignes, puis Sobel + mouvement.
`M = 0` est exactement `sobel_motion!`. `S` est un tampon image de la taille de l'instance."""
function video_rec!(out, curr, prev, S, c::NTuple{M,Any}, ab) where {M}
    H, W = size(curr)
    E = eltype(curr)
    g = one(E) - _csum(c) * one(E)    # gain unité en continu
    @inbounds for i in 1:H
        ys = ntuple(_ -> zero(E), Val(M))
        for j in 1:W
            y0 = _rstep(c, g * curr[i, j], ys)
            S[i, j] = y0
            ys = _shift(y0, ys)
        end
    end
    sobel_motion!(out, S, prev, ab)
end

"""Référence : la disposition documentée par `bench/video.jl`, soit `(H, W, nstream)`.
Chaque image est **contiguë**, donc LLVM vectorise le stencil le long de l'axe contigu. C'est
la référence FORTE — celle qui bat le DLI de 0.88× quand il n'y a pas de récurrence. Elle
appelle le même noyau que le DLI, ce qui rend la bit-exactitude exigible."""
function video_ref!(out, curr, prev, S, c::NTuple{M,Any}, ab) where {M}
    @inbounds for s in axes(curr, 3)
        video_rec!(view(out, :, :, s), view(curr, :, :, s), view(prev, :, :, s), S, c, ab)
    end
    out
end

rcoeffs(M) = ntuple(k -> T(-0.5)^k / T(3k), Val(M))

function one_order(M, nb, H, W, P)
    c = rcoeffs(M)
    cf(b, i, j) = T(sinpi(T(b)/13) + T(i)/H + T(j)/(2W))
    pf(b, i, j) = T(cospi(T(b)/11) + T(i)/(2H) + T(j)/W)

    # Même contenu, deux dispositions : (H,W,nstream) pour la référence contiguë,
    # (nstream,H,W) batch-major pour le DLI.
    curr_r = T[cf(s, i, j) for i in 1:H, j in 1:W, s in 1:nb]
    prev_r = T[pf(s, i, j) for i in 1:H, j in 1:W, s in 1:nb]
    curr_b = T[cf(s, i, j) for s in 1:nb, i in 1:H, j in 1:W]
    prev_b = T[pf(s, i, j) for s in 1:nb, i in 1:H, j in 1:W]

    Or = zeros(T, H, W, nb)
    video_ref!(Or, curr_r, prev_r, zeros(T, H, W), c, AB)

    Ci = Interleave.Array{T,3,P}(curr_b); Pi = Interleave.Array{T,3,P}(prev_b)
    Oi = similar(Ci); fill!(Oi, 0)
    Sproto = scratchlike(Ci)
    kern = (o, cu, pr, s) -> video_rec!(o, cu, pr, s, c, AB)
    apply!(kern, Oi, Ci, Pi; scratch = Sproto)
    err = maximum(abs(Oi[s, i, j] - Or[i, j, s]) for s in 1:nb, i in 1:H, j in 1:W)

    Sref = zeros(T, H, W); Oref = zeros(T, H, W, nb)
    t_ref = @belapsed video_ref!($Oref, $curr_r, $prev_r, $Sref, $c, $AB) samples=5 evals=1 seconds=4
    t_dli = @belapsed apply!($kern, $Oi, $Ci, $Pi; scratch = $Sproto) samples=5 evals=1 seconds=4
    (; t_ref, t_dli, err)
end

function run(; nb = 256, H = 128, W = 128, P = 8, orders = (0, 1, 2, 4, 8))
    println("\n", "="^80)
    println("Vidéo : Sobel + mouvement précédé d'un lissage récursif d'ordre M")
    println("$nb flux × $H×$W, P = $P, Float32")
    println("="^80)
    println("Référence = images CONTIGUËS (H,W,nstream), comme bench/video.jl.")
    println("M = 0 est exactement le noyau documenté, où le DLI perd.")
    println()
    println(rpad("M", 4), rpad("référence", 14), rpad("DLI", 14), rpad("DLI/référence", 15), "accord")
    println("-"^62)
    for M in orders
        r = one_order(M, nb, H, W, P)
        println(rpad(M, 4),
                rpad(@sprintf("%.2f ms", r.t_ref*1e3), 14),
                rpad(@sprintf("%.2f ms", r.t_dli*1e3), 14),
                rpad(@sprintf("%.2f×", r.t_ref/r.t_dli), 15),
                r.err == 0 ? "exact" : @sprintf("%.2e", r.err))
    end
    println()
    nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && run()
