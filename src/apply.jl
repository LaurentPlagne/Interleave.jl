# Les drivers : appliquent un noyau à toutes les instances du lot.
#
# `apply!` et `parallel_apply!` sont DEUX fonctions, et c'est délibéré. Une bibliothèque qui
# lance des threads en silence casse tout appelant qui se trouve déjà dans une région
# parallèle. `apply!` ne peut pas paralléliser — il n'a aucun mot-clé pour ça — de sorte
# que le parallélisme est toujours visible au point d'appel. C'est la raison d'être du
# `parmap` explicite de Legolas++.
#
# Aucune vectorisation explicite ici, et surtout aucun `@fastmath` : l'invariant 1
# (bit-exactitude scalaire ↔ vectorisé) l'interdit. Cf. AGENTS.md.

# Aucune annotation abstraite ici : `Tuple` nu, comme `Function`, empêche la
# spécialisation et rend l'appel dynamique. Tout est paramétré — `F` pour le noyau,
# `NA` pour l'arité, ce qui force Julia à compiler une instance par combinaison.
@inline _apply(f::F, views::NTuple{NA,Any}, ::Nothing) where {F,NA} = f(views...)
@inline _apply(f::F, views::NTuple{NA,Any}, scratch) where {F,NA} = f(views..., scratch)

"""Build the fixed-arity tuple of instance views without a closure allocation."""
@generated function _instances(arrays::NTuple{NA,Any}, k) where {NA}
    Expr(:tuple, [:(instance(arrays[$i], k)) for i in 1:NA]...)
end

function _runpacks!(f::F, ks, arrays::NTuple{NA,Any}, scratch) where {F,NA}
    for k in ks
        _apply(f, _instances(arrays, k), scratch)
    end
    nothing
end

# `scratch` accepte un **prototype de tableau** — le driver en fait un `similar` par
# chunk, de sorte que chaque tâche ait le sien. Pas besoin d'un type maison : `similar`
# de Base a exactement le bon contrat. Un appelable sans argument reste accepté, pour les
# cas où l'on veut maîtriser l'allocation à la main.
@inline _newscratch(::Nothing) = nothing
@inline _newscratch(proto::AbstractArray) = similar(proto)
@inline _newscratch(f) = f()

# `size(A, 1)` est le nombre d'instances pour un `Interleave.Array` comme pour un
# `Base.Array` : c'est la taille logique du lot dans les deux cas.
function _checked_npacks(arrays::NTuple{NA,Any}) where NA
    A = first(arrays)
    for B in arrays
        size(B, 1) == size(A, 1) ||
            throw(DimensionMismatch("batch sizes differ: $(size(A,1)) and $(size(B,1))"))
        packsize(B) == packsize(A) ||
            throw(DimensionMismatch(
                "packet sizes differ: $(packsize(A)) and $(packsize(B))"))
    end
    npacks(A)
end

"""
    apply!(f, arrays...; scratch = nothing)

Apply kernel `f` to every problem in the batch, **sequentially**. SIMD comes from the
machine element type rather than task parallelism.

Both [`Interleave.Array`](@ref) and ordinary `Base.Array` batches are accepted. For a standard
array, the first dimension is the batch and each packet contains one lane, which makes it a
convenient development and reference path.

`f` receives one instance view per array. Packed instances are dense; the reference
`Base.Array` view may be strided because the batch is its first dimension. If `scratch` is
supplied, it is appended as the last argument.

!!! warning "Never launches tasks"
    `apply!` cannot parallelize: it has no scheduler or chunk-count keyword. Use
    [`parallel_apply!`](@ref) when task creation should be explicit at the call site.

- `scratch`: an array prototype copied with `similar`, commonly
  `similar(instance(X, 1))`. A zero-argument callable is also accepted for custom
  construction. Scratch storage is never allocated inside the kernel.

Return the first array.

# Example
```julia
apply!(thomas!, X, D, U, L, B; scratch = similar(instance(X, 1)))
```
"""
function apply!(f::F, arrays::Vararg{AbstractArray,NA}; scratch = nothing) where {F,NA}
    npk = _checked_npacks(arrays)
    _runpacks!(f, 1:npk, arrays, _newscratch(scratch))
    first(arrays)
end

"""
    parallel_apply!(f, arrays...; scratch = nothing, scheduler = StaticScheduler(), nchunks = ...)

Like [`apply!`](@ref), but distribute packet chunks across tasks. Call it only when the
surrounding context does not already own parallelism; the separate name keeps that decision
visible.

- `scheduler`: an OhMyThreads scheduler (`StaticScheduler`, `DynamicScheduler`,
  `GreedyScheduler`, or `SerialScheduler`). Use `chunking=false` when exactly one task per
  chunk is required.
- `scratch`: an array prototype copied with `similar` **once per chunk**, so every task owns
  its mutable workspace. Sharing one scratch object between tasks creates a race.
- `nchunks`: packet-partition granularity.

Return the first array.
"""
function parallel_apply!(f::F, arrays::Vararg{AbstractArray,NA};
                   scratch = nothing,
                   scheduler = StaticScheduler(),
                   nchunks::Int = 4 * Threads.nthreads()) where {F,NA}
    npk = _checked_npacks(arrays)
    if scheduler isa SerialScheduler
        _runpacks!(f, 1:npk, arrays, _newscratch(scratch))
    else
        chunks = index_chunks(1:npk; n = max(1, min(nchunks, npk)))
        tforeach(chunks; scheduler) do ks
            _runpacks!(f, ks, arrays, _newscratch(scratch))
        end
    end
    first(arrays)
end
