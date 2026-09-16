module InterleaveSerializationExt

using Interleave: Interleave
using Serialization: Serialization, AbstractSerializer, serialize_type

# `Interleave.Array` garde un stockage paqueté `data` et une vue scalaire `flat` qui
# **alias la même mémoire** via `unsafe_wrap` (array.jl, invariant 4). `Serialization`
# reconstruit une struct **champ par champ** : après un aller-retour, `data` et `flat`
# seraient deux tampons indépendants. L'indexation scalaire écrirait alors dans un buffer
# qu'aucun noyau ne lit — résultat faux, et silencieux.
#
# C'est exactement le danger que `Base.deepcopy_internal` traite dans `src/array.jl`. Le
# remède est le même : ne sérialiser que le stockage paqueté et la taille du lot, et
# laisser le **constructeur** rester le seul endroit qui établit l'alias.
#
# ⚠️ Tout autre sérialiseur reconstruisant la struct champ par champ (JLD2, BSON, Arrow)
# reproduit le bug et demande le même traitement. Le champ `flat` est la raison de cette
# fragilité ; il est conservé parce qu'il vaut 3.8× le `reinterpret` sur l'accès scalaire
# (AGENTS.md §2 ter).

function Serialization.serialize(s::AbstractSerializer,
                                 A::Interleave.Array{T,N,P,M,E}) where {T,N,P,M,E}
    serialize_type(s, typeof(A))
    Serialization.serialize(s, A.data)
    Serialization.serialize(s, A.nbatch)
    nothing
end

function Serialization.deserialize(s::AbstractSerializer,
                                   ::Type{Interleave.Array{T,N,P,M,E}}) where {T,N,P,M,E}
    data = Serialization.deserialize(s)::Base.Array{E,N}
    nbatch = Serialization.deserialize(s)::Int
    Interleave._wrap(T, Val(P), data, nbatch)
end

end # module
