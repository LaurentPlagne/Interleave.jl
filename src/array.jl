# Data Layout Interleaving (DLI) : `nbatch` instances indépendantes d'un même problème,
# entrelacées par paquets de `P` lanes.
#
# Stockage natif = `Base.Array{Vec{P,T},N}` dense de taille `(dims_instance..., npacks)`.
# C'est l'inverse du choix C++ (qui stocke des scalaires et dérive une vue packée) :
# ici le chemin chaud manipule un tableau dense ordinaire. Une vue scalaire interne du même
# buffer sert au remplissage et aux I/O, mais elle n'entre jamais dans le chemin du noyau.
# Voir AGENTS.md, invariant 4.

"""
    packtype(T, Val(P)) -> Type

Machine element type for a batch interleaved in packets of `P`: **`T` itself when
`P == 1`**, and `Vec{P,T}` otherwise.

The scalar special case matters. `Vec{1,T}` can prevent LLVM from vectorizing an ordinary
contiguous loop. Returning `T` makes `P=1` the normal dense-array regime, so one kernel can
cover both recurrence-heavy and already-vectorizable workloads.
"""
@inline packtype(::Type{T}, ::Val{1}) where {T} = T
@inline packtype(::Type{T}, ::Val{P}) where {T,P} = Vec{P,T}

"""
    Interleave.Array{T,N,P}

An `N`-dimensional scalar array whose **first dimension** is stored in packets of `P` using
Data Layout Interleaving.

The first two parameters have the same meaning as for `Base.Array`: scalar element type `T`
and number of dimensions `N`. `P` is the only addition, allowing the storage type to be
changed in one line:

```julia
const Arr = Base.Array{Float32,2}        # development
const Arr = Interleave.Array{Float32,2,8}   # packed production layout
X = Arr(undef, nsys, nx)                 # identical construction
```

The first dimension is the batch: `A` contains `size(A, 1)` independent problems, each with
shape `size(A)[2:end]`. [`apply!`](@ref) invokes a kernel on those problems.

A `Interleave.Array` **is logically an array of scalars**:
`size(A) == (nbatch, instance_dims...)`, `eltype(A) === T`, and `A[b, i, j]` returns a
scalar. Fill, read, reduce, and compare it like any other `AbstractArray`.

The machine representation is available through `parent(A)`: a dense array of packed
elements with shape `(instance_dims..., npacks)`. The driver slices this storage for the
kernel. Normal user code does not need it.

`P` fixes the machine element type (see [`packtype`](@ref)) and can change performance, but
never changes the logical semantics.

# Example
```julia
D = Array{Float32}(undef, 65_536, 64; pack = Val(8))
D .= 2.0f0                 # ordinary broadcast with scalar indices
D[7, 3] = 1.5f0            # scalar indexing
apply!(kernel!, D)         # the kernel receives packet views
```
"""
struct Array{T,N,P,M,E} <: AbstractArray{T,N}
    data::Base.Array{E,N}
    flat::Base.Array{T,M}          # (P, dims_instance..., npacks), MÊME mémoire que `data`
    nbatch::Int

    function Array{T,N,P,M,E}(data::Base.Array{E,N}, nbatch::Integer) where {T,N,P,M,E}
        # Invariant 2 : sizeof(Vec{P,T}) == P*sizeof(T) n'est garanti que pour P
        # puissance de deux ; toute l'indexation scalaire en dépend.
        P isa Integer && ispow2(P) ||
            throw(ArgumentError("packet size P must be a power of two, got P=$P"))
        E === packtype(T, Val(P)) ||
            throw(ArgumentError("element type $E is incompatible with T=$T and P=$P"))
        sizeof(E) == P * sizeof(T) ||
            throw(ArgumentError("$E is padded to $(sizeof(E)) bytes; cannot build a DLI layout"))
        N ≥ 2 || throw(ArgumentError("an instance must have at least one dimension"))
        0 ≤ nbatch ≤ typemax(Int) ||
            throw(ArgumentError("batch size must be nonnegative, got nbatch=$nbatch"))
        cld(nbatch, P) == size(data, N) ||
            throw(DimensionMismatch(
                "nbatch=$nbatch with P=$P requires $(cld(nbatch,P)) packets, " *
                "but the storage contains $(size(data,N))"))
        M == N + 1 || throw(ArgumentError("M must equal N+1"))
        # Vue scalaire du MÊME buffer, en vrai `Array` et non en `ReinterpretArray` :
        # mesuré 3.8× plus rapide sur un stencil `@simd` (voir AGENTS.md §2 ter).
        # `flat` n'est pas propriétaire ; c'est `data`, champ de la même structure, qui
        # maintient la mémoire en vie tant que le conteneur `A` reste accessible. Cette
        # vue est strictement interne et ne doit donc pas lui survivre.
        flat = unsafe_wrap(Base.Array, reinterpret(Ptr{T}, pointer(data)), (P, size(data)...))

        # Centraliser l'initialisation ici couvre aussi les tableaux construits depuis
        # un stockage paqueté existant et ceux créés par `similar`.
        npk = size(data, N)
        nvalid = npk == 0 ? P : nbatch - (npk - 1) * P
        if nvalid < P
            padding = view(flat, nvalid + 1:P,
                           ntuple(_ -> Colon(), Val(N - 1))..., npk)
            fill!(padding, zero(T))
        end
        new{T,N,P,M,E}(data, flat, Int(nbatch))
    end
end

# `Array{T,N}` signifierait « N dimensions », comme chez Base : on ne définit donc aucun
# constructeur à deux paramètres. L'assemblage interne passe par `_wrap`.
_wrap(::Type{T}, ::Val{P}, data::Base.Array{E,N}, nbatch::Integer) where {T,P,E,N} =
    Array{T,N,P,N + 1,E}(data, nbatch)
Array(data::Base.Array{Vec{P,T},N}, nbatch::Integer) where {T,P,N} =
    _wrap(T, Val(P), data, nbatch)

# `P` est un **paramètre de type**, et non un mot-clé : c'est ce qui permet de basculer
# d'un tableau standard à un tableau entrelacé en changeant une seule ligne — un alias.
#
#     const Arr = Base.Array{Float32,2}      # mise au point
#     const Arr = Interleave.Array{Float32,2,8} # production
#     X = Arr(undef, nsys, nx)               # construction identique dans les deux cas
#
# La forme `Array{T}(undef, …; pack = Val(P))` reste disponible quand `P` est calculé.
function Array{T,N,P}(::UndefInitializer, nbatch::Integer, dims::Vararg{Integer,ND}) where {T,N,P,ND}
    ND + 1 == N ||
        throw(ArgumentError("Array{$T,$N,$P} expects $(N-1) instance dimensions, got $ND"))
    P isa Integer && ispow2(P) ||
        throw(ArgumentError("packet size P must be a power of two, got P=$P"))
    0 ≤ nbatch ≤ typemax(Int) ||
        throw(ArgumentError("batch size must be nonnegative, got nbatch=$nbatch"))
    npk = cld(nbatch, P)
    E = packtype(T, Val(P))
    data = Base.Array{E}(undef, dims..., npk)
    _wrap(T, Val(P), data, nbatch)
end

Array{T}(u::UndefInitializer, nbatch::Integer, dims::Integer...;
         pack::Val{P} = Val(4)) where {T,P} =
    Array{T,length(dims) + 1,P}(u, nbatch, dims...)

"""
    Interleave.Array{T,N,P}(source::AbstractArray)
    Interleave.Array(source; pack = Val(4))

Copy a scalar array into a Data Layout Interleaved array while preserving its logical
shape. This makes an ordinary Julia comprehension a convenient initializer:

```julia
X = Interleave.Array{Float32,2,8}([
    Float32(job) + Float32(i) / 32 for job in 1:61, i in 1:128
])
```

The comprehension is materialized before it is packed, so use the `undef` constructor
followed by broadcast or a loop when initialization cost or peak memory matters.
"""
function Array{T,N,P}(source::AbstractArray{S,N}) where {T,N,P,S}
    dest = Array{T,N,P}(undef, size(source)...)
    copyto!(dest, source)
    dest
end

Array(source::AbstractArray{T,N}; pack::Val{P} = Val(4)) where {T,N,P} =
    Array{T,N,P}(source)

"""
    lanetype(x) -> Type

Underlying **scalar** type of a batch element: `Float32` for both `Float32` and
`Vec{P,Float32}`.

Use it to construct a correctly typed scalar literal in a generic kernel. A scalar of the
right type can multiply either a scalar or a SIMD packet without relying on conversion from
a default `Float64` literal.

```julia
S = lanetype(eltype(x))
y = v * S(0.125)
```
"""
lanetype(::Type{Vec{P,T}}) where {P,T} = T
lanetype(::Type{T}) where {T<:Number} = T
lanetype(x) = lanetype(typeof(x))

"""Number of lanes in one packet (the layout parameter `P`)."""
packsize(::Array{T,N,P}) where {T,N,P} = P
packsize(::Type{<:Array{T,N,P}}) where {T,N,P} = P

"""Number of packets, equal to `cld(A.nbatch, packsize(A))`."""
npacks(A::Array{T,N}) where {T,N} = size(A.data, N)

"""Shape of one problem instance, without the packet dimension."""
instance_size(A::Array) = Base.front(size(A.data))

"""
    packtype(A) -> Type

**Machine** element type seen by a kernel. It is `T` when `P == 1` and `Vec{P,T}`
otherwise. This differs from the logical `eltype(A)`, which is always scalar.

Useful when constructing scratch storage directly. In most cases,
`similar(instance(A, 1))` is sufficient.
"""
packtype(A::Array{T,N,P}) where {T,N,P} = packtype(T, Val(P))

"""Number of non-significant padding lanes (`npacks*P - nbatch`)."""
npadding(A::Array) = npacks(A) * packsize(A) - A.nbatch

# Interface `AbstractArray` : c'est la vue **logique** qui est exposée — `(nbatch, dims…)`
# d'éléments **scalaires**. L'entrelacement n'apparaît nulle part dans l'indexation :
# `A[b, i, j]` rend un `T`, pas un paquet.
Base.size(A::Array) = (A.nbatch, Base.front(Base.tail(size(A.flat)))...)
Base.IndexStyle(::Type{<:Array}) = IndexCartesian()

# `Vararg{Int,N}` avec N = ndims : la méthode ne capte QUE l'indexation cartésienne
# complète. L'indexation linéaire (`A[17]`) reste au repli `IndexCartesian` de Base, qui
# la convertit en indices cartésiens — sans cette contrainte, `A[17]` serait pris pour un
# numéro d'instance et rendrait silencieusement le mauvais élément.
Base.@propagate_inbounds function Base.getindex(A::Array{T,N,P},
                                                I::Vararg{Int,N}) where {T,N,P}
    @boundscheck checkbounds(A, I...)
    k, p = divrem(first(I) - 1, P)
    @inbounds A.flat[p+1, Base.tail(I)..., k+1]
end

Base.@propagate_inbounds function Base.setindex!(A::Array{T,N,P}, x,
                                                 I::Vararg{Int,N}) where {T,N,P}
    @boundscheck checkbounds(A, I...)
    k, p = divrem(first(I) - 1, P)
    @inbounds A.flat[p+1, Base.tail(I)..., k+1] = x
end

"""
    parent(A) -> Array

The batch's **machine representation**: a dense array of packed elements with shape
`(instance_dims..., npacks)`. [`apply!`](@ref) and [`parallel_apply!`](@ref) slice this
storage; `A` itself remains scalar-indexed.
"""
Base.parent(A::Array) = A.data

_packable(::Type{S}, ::Val{1}) where {S} = isbitstype(S)
_packable(::Type{S}, ::Val{P}) where {S,P} = S <: VecTypes

function _similar(::Type{S}, p::Val{P}, dims::Dims{N}) where {S,P,N}
    N ≥ 2 && _packable(S, p) ? Array{S,N,P}(undef, dims...) : Base.Array{S}(undef, dims)
end

Base.similar(A::Array{T,N,P}) where {T,N,P} = _similar(T, Val(P), size(A))
Base.similar(A::Array{T,N,P}, ::Type{S}) where {T,N,P,S} =
    _similar(S, Val(P), size(A))
Base.similar(A::Array{T,N,P}, dims::Dims) where {T,N,P} =
    _similar(T, Val(P), dims)
Base.similar(A::Array{T,N,P}, ::Type{S}, dims::Dims) where {T,N,P,S} =
    _similar(S, Val(P), dims)
Base.fill!(A::Array{T,N,P}, x) where {T,N,P} =
    (fill!(A.data, packtype(T, Val(P))(convert(T, x))); A)

"""
    instance(A, k)

View of the `k`th packet of problems: a dense `SubArray` with the shape of one instance and
machine element type `Vec{P,T}` (or `T` for a standard array). This is what the kernel
receives; the driver owns batch traversal.
"""
@inline instance(A::Array{T,N}, k::Integer) where {T,N} =
    view(A.data, ntuple(_ -> Colon(), Val(N - 1))..., k)

# ---------------------------------------------------------------------------------
# Tableaux standard
#
# Un `Base.Array` est un lot valide : sa première dimension joue le rôle du lot, et il n'y
# a qu'une lane par paquet. On peut donc **écrire et tester un algorithme avec un tableau
# standard**, puis ne changer que le type pour gagner en vitesse — c'est le principe de
# Legolas++, où `Legolas::Array<T,D>` est déjà un tableau rectangulaire ordinaire et où le
# packing n'est qu'un jeu de paramètres supplémentaires.
#
# Le prix du tableau standard est la localité : `view(A, k, :, :)` est **fractionné** en
# ordre colonne-majeur, là où une instance d'un `Interleave.Array` est contiguë. C'est
# précisément ce que l'entrelacement corrige.

packsize(::AbstractArray) = 1
npacks(A::AbstractArray) = size(A, 1)
instance_size(A::AbstractArray) = Base.tail(size(A))
npadding(::AbstractArray) = 0
packtype(A::AbstractArray) = eltype(A)

@inline instance(A::AbstractArray{T,N}, k::Integer) where {T,N} =
    view(A, k, ntuple(_ -> Colon(), Val(N - 1))...)

# ---------------------------------------------------------------------------------
# Itérateur de paquets
#
# `packs(A)` est un `AbstractVector` dont les **éléments sont les vues de paquet**. Il
# donne à `Base.foreach` exactement la bonne sémantique : effet de bord, rend `nothing`,
# élémentwise sur plusieurs itérateurs.
#
# La méthode spécialisée de `foreach` ci-dessous n'est pas de la piraterie de type :
# `Packs` nous appartient. Sans elle, le `zip` générique de `Base` fait échapper les vues
# sur le tas (mesuré : 9,4 Mo là où la version spécialisée n'alloue rien).

struct Packs{A,V} <: AbstractVector{V}
    array::A
    n::Int
end

"""
    packs(A) -> AbstractVector

A lazy vector whose elements are the packet views of `A`—the values received by a kernel.
It supports sequential traversal with `Base.foreach`:

```julia
foreach(kernel!, packs(X), packs(D), packs(U))
```

`foreach` is **sequential**, like [`apply!`](@ref). It has no per-task scratch storage or
scheduler; use [`parallel_apply!`](@ref) for those features.
"""
function packs(A::AT) where {AT<:AbstractArray}
    # Le type de la vue d'instance ne dépend que du type de `A`, pas de sa taille.
    # `promote_op` l'obtient sans indexer un hypothétique premier paquet, ce qui rend
    # les lots vides valides tout en conservant un `Packs` entièrement concret.
    V = Base.promote_op(instance, AT, Int)
    Packs{AT,V}(A, npacks(A))
end

Base.size(p::Packs) = (p.n,)
Base.IndexStyle(::Type{<:Packs}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(p::Packs, k::Int) = instance(p.array, k)

function Base.foreach(f::F, a::Packs, rest::Vararg{Packs,NR}) where {F,NR}
    n = length(a)
    all(r -> length(r) == n, rest) ||
        throw(DimensionMismatch("packet iterators have different lengths"))
    @inbounds for k in 1:n
        f(a[k], ntuple(i -> rest[i][k], Val(NR))...)
    end
    nothing
end
