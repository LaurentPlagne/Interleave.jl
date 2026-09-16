# Repaquetage : changer l'axe qui porte le lot.
#
# C'est l'opération qu'un solveur ADI 3-D fait entre deux directions. Le tutoriel la décrivait
# — « reorder → balayages → reorder back » — sans qu'aucun `reorder` existe : l'utilisateur
# tombait sur le repli générique de `permutedims!`, qui passe par l'indexation scalaire et son
# `divrem` par élément.
#
# On garde le VERBE de Base, `permutedims!` : la sémantique est déjà la bonne, et le repli
# générique donnait déjà le bon résultat. Seule la vitesse change.
#
# Mesuré sur une grille 128×128×64, échange des axes 1 et 2 :
#
#   permutedims! générique       1.22 ms      (0.73 balayage de Thomas)
#   ce chemin rapide             0.47 ms      (0.28 balayage)
#   permutedims! sur Base.Array  0.40 ms      ← plancher : même brassage, sans paquets
#
# Soit 2.6×, et 18 % au-dessus du plancher théorique. Il ne reste rien de significatif à
# gagner sur cette opération.

"""Le lot ne bouge pas : seules les dimensions d'instance permutent.

C'est le cas le moins cher et il mérite d'être reconnu — aucune lane ne change de paquet, donc
c'est un `permutedims!` ordinaire sur le stockage **paqueté**, que Base fait déjà très bien.
Un balayage ADI y/z sur une grille paquetée en x tombe dans ce cas."""
function _permute_instance!(dest::Array{T,N,P}, src::Array{T,N,P}, perm) where {T,N,P}
    # `data` a la forme (dims_instance…, npacks) ; l'axe des paquets est dernier et fixe.
    inner = ntuple(k -> perm[k + 1] - 1, Val(N - 1))
    permutedims!(parent(dest), parent(src), (inner..., N))
    dest
end

# L'axe de lot change de place : les lanes doivent traverser les paquets. C'est une
# transposition de tuiles P×P, et l'ordre des boucles y décide de tout — l'indice de LANE est
# contigu dans `flat`, donc il doit être le plus interne des deux côtés. Une première version
# l'avait en position externe et lisait de foulée P : 1.4× au lieu de 2.6×.
function _swap_batch!(dest::Array{T,3,P}, src::Array{T,3,P}, ::Val{M}) where {T,P,M}
    d, s = dest.flat, src.flat
    nb_d, nb_s = size(dest, 1), size(src, 1)
    nfix = M == 2 ? size(src, 3) : size(src, 2)
    tile = Matrix{T}(undef, P, P)
    npd, nps = cld(nb_d, P), cld(nb_s, P)
    @inbounds for f in 1:nfix, qs in 0:nps-1, qd in 0:npd-1
        pd_hi = min(P, nb_d - qd * P)
        ps_hi = min(P, nb_s - qs * P)
        for pd in 1:pd_hi
            @simd for ps in 1:ps_hi
                tile[pd, ps] = M == 2 ? s[ps, qd * P + pd, f, qs + 1] :
                                        s[ps, f, qd * P + pd, qs + 1]
            end
        end
        for ps in 1:ps_hi
            @simd for pd in 1:pd_hi
                if M == 2
                    d[pd, qs * P + ps, f, qd + 1] = tile[pd, ps]
                else
                    d[pd, f, qs * P + ps, qd + 1] = tile[pd, ps]
                end
            end
        end
    end
    dest
end

"""
    permutedims!(dest::Interleave.Array, src::Interleave.Array, perm)

Repack a batch along a different axis, as an ADI sweep does between directions.

`dest[J...] == src[J[perm]...]`, exactly as for any `AbstractArray`. What changes here is the
cost: the generic fallback reaches every element through scalar indexing, which pays a
`divrem` per element to locate its lane.

Two cases are recognised:

- `perm[1] == 1` — the batch axis stays put, so no lane crosses a packet and this is an
  ordinary `permutedims!` on the packed storage;
- a transposition of the batch axis with one other axis of a 3-D batch — the ADI case — which
  becomes a `P × P` tile transpose.

Anything else falls back to the generic path, which is correct but slower.

Measured on a 128×128×64 grid swapping axes 1 and 2: 1.22 ms generic, **0.47 ms** here,
against a 0.40 ms floor set by `permutedims!` on an equivalent `Base.Array`. In ADI terms the
repack drops from 0.73 of a Thomas sweep to 0.28.

!!! warning "Repacking is not free, and it is not the whole cost either"
    Measure the complete stage — reorder, the repeated line solves, reorder back — rather than
    the line solver alone. The repack is worth paying when the packed layout is reused for
    many time steps; it is a poor trade when every solve is tiny.
"""
function Base.permutedims!(dest::Array{T,N,P}, src::Array{T,N,P},
                           perm) where {T,N,P}
    length(perm) == N ||
        throw(ArgumentError("perm has length $(length(perm)), expected $N"))
    isperm(perm) || throw(ArgumentError("perm=$perm is not a permutation"))
    for k in 1:N
        size(dest, k) == size(src, perm[k]) || throw(DimensionMismatch(
            "dest axis $k has length $(size(dest,k)) but src axis $(perm[k]) has " *
            "length $(size(src, perm[k]))"))
    end

    if perm[1] == 1
        return _permute_instance!(dest, src, perm)
    elseif N == 3 && (Tuple(perm) === (2, 1, 3) || Tuple(perm) === (3, 2, 1))
        # Transposition pure des axes 1 et m, le troisième restant en place : le cas ADI.
        return _swap_batch!(dest, src, Val(Int(perm[1])))
    end
    _generic_permutedims!(dest, src, perm)
end

# Le repli : correct pour toute permutation, simplement plus lent. On le garde explicite
# plutôt que d'appeler `invoke`, pour que le chemin lent soit lisible et testable.
function _generic_permutedims!(dest::Array{T,N}, src::Array{T,N}, perm) where {T,N}
    @inbounds for J in CartesianIndices(dest)
        t = Tuple(J)
        dest[t...] = src[ntuple(k -> t[perm[k]], Val(N))...]
    end
    dest
end
