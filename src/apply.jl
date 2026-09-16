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

"""Build the fixed-arity tuple of packet views without a closure allocation."""
@generated function _packets(arrays::NTuple{NA,Any}, k) where {NA}
    Expr(:tuple, [:(packet(arrays[$i], k)) for i in 1:NA]...)
end

function _runpacks!(f::F, ks, arrays::NTuple{NA,Any}, scratch) where {F,NA}
    for k in ks
        _apply(f, _packets(arrays, k), scratch)
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

# Sans cette vérification, un noyau appelé avec la mauvaise arité produit une `MethodError`
# dont la signature est illisible : cinq `SubArray{Vec{8,Float32},1,Matrix{Vec{8,Float32}},
# Tuple{Base.Slice{Base.OneTo{Int64}},Int64},true}` alignés, et la vraie cause — « il manque
# le scratch » — n'apparaît que dans un `!Matched::Any` en fin de liste. `applicable` coûte
# un appel par `apply!`, jamais par paquet, donc le chemin chaud est intact.
# `nameof(typeof(f))` rend le nom mangle (`#thomas!`) ; `nameof(f)` rend `thomas!`, mais
# n'existe que pour une `Function` — un type appelable passe par son type.
# Un noyau anonyme rend `#6`, qui n'aide personne : on dit simplement « the kernel ».
_kernel_name(f::Function) = _kernel_ref(string(nameof(f)))
_kernel_name(f) = _kernel_ref(string(nameof(typeof(f))))
_kernel_ref(n) = startswith(n, '#') ? "the kernel" : "kernel `$n`"

@noinline function _kernel_arity_error(f, arrays::NTuple{NA,Any}, scratch) where {NA}
    views = _packets(arrays, 1)
    name = _kernel_name(f)
    plural = NA == 1 ? "" : "s"
    if scratch === nothing
        # Le noyau accepterait-il un argument de plus ? C'est le cas le plus fréquent.
        if applicable(f, views..., scratchlike(first(arrays)))
            throw(ArgumentError(
                "$name cannot be called with $NA packet view$plural and no " *
                "workspace, but it accepts one more argument. It most likely needs a " *
                "scratch buffer:\n" *
                "    apply!(kernel, arrays...; scratch = scratchlike(first_array))"))
        end
    else
        # Symétrique : un scratch passé à un noyau qui n'en veut pas.
        if applicable(f, views...)
            throw(ArgumentError(
                "$name takes $NA packet view$plural and no workspace, but a " *
                "`scratch` argument was supplied. Drop the `scratch` keyword."))
        end
    end
    throw(ArgumentError(
        "$name cannot be called on this batch. It is invoked with $NA packet " *
        "view$plural of element type $(eltype(first(views)))" *
        (scratch === nothing ? "" : " plus a workspace of element type $(eltype(scratch))") *
        ".\nA kernel must accept one view per array passed to the driver, in the same " *
        "order, and the workspace last when `scratch` is given."))
end

@inline function _check_kernel(f::F, arrays::NTuple{NA,Any}, scratch) where {F,NA}
    npacks(first(arrays)) == 0 && return nothing
    views = _packets(arrays, 1)
    ok = scratch === nothing ? applicable(f, views...) : applicable(f, views..., scratch)
    ok || _kernel_arity_error(f, arrays, scratch)
    nothing
end

# Le scratch CPU a la taille d'**une** instance ; le scratch GPU est un lot batch-major.
# Les confondre est l'erreur documentée dans docs/src/manual/review.md, et elle produisait
# jusqu'ici un `convert` illisible venu d'un paquet sans rapport.
@noinline function _scratch_shape_error(arrays::NTuple{NA,Any}, scratch) where {NA}
    A = first(arrays)
    throw(DimensionMismatch(
        "scratch has size $(size(scratch)), but `apply!` expects one instance of " *
        "workspace, of size $(instance_size(A)).\n" *
        "It looks like a batch-major buffer, which is what `gpu_apply!` wants. " *
        "For the CPU driver use `scratch = scratchlike(A)`."))
end

@inline function _check_scratch(arrays::NTuple{NA,Any}, scratch) where {NA}
    scratch isa AbstractArray || return nothing
    A = first(arrays)
    size(scratch) == instance_size(A) && return nothing
    # Seul le cas franchement batch-major est refusé ; un scratch plus grand reste permis.
    ndims(scratch) == ndims(A) && size(scratch, 1) == size(A, 1) &&
        _scratch_shape_error(arrays, scratch)
    nothing
end

# Un noyau peut être parfaitement APPELABLE et violer quand même le contrat. Le cas courant
# est le branchement sur une donnée : à `P > 1`, `x[i] > 0` rend un `Vec{P,Bool}`, pas un
# `Bool`, et un `if` dessus n'a plus de sens — les lanes ne sont pas d'accord entre elles.
#
# Julia dit alors « non-boolean used in boolean context », ce qui est exact mais muet sur la
# marche à suivre. Pire : la réponse réflexe, `ifelse`, n'a pas non plus de méthode pour `Vec`.
# L'utilisateur est donc dans une impasse sans indication. La sortie est `vifelse`, qui marche
# sur un `Vec` ET sur un scalaire — donc le même noyau reste valide à `P = 1`.
#
# Le `try` ne coûte rien quand rien n'est levé ; il n'entoure pas le corps du noyau mais la
# boucle entière, donc le chemin chaud est intact.
const _CONTRACT_HINT = """
A kernel must use the SAME control flow for every lane of a packet. Rewrite the branch
without one:

    y[i] = x[i] > 0 ? x[i] : -x[i]        # invalid: the test is a Vec{P,Bool}
    y[i] = vifelse(x[i] > 0, x[i], -x[i]) # valid, and still correct at P = 1

`vifelse` is re-exported by Interleave. What cannot be expressed this way at all is a
data-dependent early exit, a data-dependent index, or per-lane recursion depth; those kernels
need `P = 1`, or a different decomposition."""

@noinline function _rethrow_kernel_error(f, e)
    name = _kernel_name(f)
    if e isa TypeError && e.expected === Bool
        throw(ArgumentError(
            "$name branches on data: a comparison between packets yields a " *
            "`Vec{P,Bool}`, which has no single truth value.\n\n" * _CONTRACT_HINT))
    elseif e isa MethodError && any(a -> a isa Vec, e.args)
        throw(ArgumentError(
            "$name applies `$(e.f)` to a packet, and no method accepts one. " *
            "Scalar-only operations — conversion to `Int`, indexing by a value, branching — " *
            "do not lift to `Vec{P,T}`.\n\n" * _CONTRACT_HINT))
    end
    rethrow(e)
end

@inline function _guarded(f::F, body::G) where {F,G}
    try
        body()
    catch e
        _rethrow_kernel_error(f, e)
    end
end

"""
    apply!(f, arrays...; scratch = nothing)

Apply kernel `f` to every problem in the batch, **sequentially**. SIMD comes from the
machine element type rather than task parallelism.

Both [`Interleave.Array`](@ref) and ordinary `Base.Array` batches are accepted. For a standard
array, the first dimension is the batch and each packet contains one lane, which makes it a
convenient development and reference path.

`f` receives one **packet** view per array (see [`packet`](@ref)). Packed views are dense; the
reference `Base.Array` view may be strided because the batch is its first dimension. If
`scratch` is supplied, it is appended as the last argument.

!!! warning "Never launches tasks"
    `apply!` cannot parallelize: it has no scheduler or chunk-count keyword. Use
    [`parallel_apply!`](@ref) when task creation should be explicit at the call site.

- `scratch`: **one instance** of workspace, obtained with [`scratchlike`](@ref). It is copied
  with `similar`, so a zero-argument callable is also accepted for custom construction.
  Scratch storage is never allocated inside the kernel. This is *not* the batch-major buffer
  that [`gpu_apply!`](@ref) expects; passing one is rejected with an explicit message.

Return the first array.

# Example
```julia
apply!(thomas!, X, D, U, L, B; scratch = scratchlike(X))
```
"""
function apply!(f::F, arrays::Vararg{AbstractArray,NA}; scratch = nothing) where {F,NA}
    npk = _checked_npacks(arrays)
    _check_scratch(arrays, scratch)
    sc = _newscratch(scratch)
    _check_kernel(f, arrays, sc)
    _guarded(f, () -> _runpacks!(f, 1:npk, arrays, sc))
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
- `scratch`: an instance-sized prototype from [`scratchlike`](@ref), copied with `similar`
  **once per chunk**, so every task owns its mutable workspace. Sharing one scratch object
  between tasks creates a race.
- `nchunks`: packet-partition granularity.

Return the first array.
"""
function parallel_apply!(f::F, arrays::Vararg{AbstractArray,NA};
                   scratch = nothing,
                   scheduler = StaticScheduler(),
                   nchunks::Int = 4 * Threads.nthreads()) where {F,NA}
    npk = _checked_npacks(arrays)
    _check_scratch(arrays, scratch)
    _check_kernel(f, arrays, _newscratch(scratch))
    if scheduler isa SerialScheduler
        _guarded(f, () -> _runpacks!(f, 1:npk, arrays, _newscratch(scratch)))
    else
        chunks = index_chunks(1:npk; n = max(1, min(nchunks, npk)))
        tforeach(chunks; scheduler) do ks
            _runpacks!(f, ks, arrays, _newscratch(scratch))
        end
    end
    first(arrays)
end
