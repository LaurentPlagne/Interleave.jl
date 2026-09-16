# Choix de `P` par la mesure.
#
# Toute la documentation répète que `P` se règle **par noyau et par machine** — et la
# bibliothèque n'offrait aucun moyen de le faire. Chaque utilisateur devait reconstruire le
# harnais qui vit dans `bench/`, hors d'atteinte du paquet installé. `tune` est ce harnais,
# réduit à ce qui sert à décider.
#
# Pas de dépendance à BenchmarkTools : un minimum sur quelques tours suffit à départager des
# variantes qui diffèrent d'un facteur 2 à 15. Ce n'est pas un remplaçant de `@benchmark`
# pour une mesure publiable, et la docstring le dit.

"""
    TuningResult

Result of [`tune`](@ref): the measured packet sizes, the best wall time of each, and the one
that won. `show` renders it as a table with speedups relative to `P = 1` when that size was
measured.
"""
struct TuningResult
    packs::Vector{Int}
    times::Vector{Float64}
    best::Int
    kernel::String
end

Base.getindex(r::TuningResult, P::Integer) = r.times[findfirst(==(Int(P)), r.packs)]

function Base.show(io::IO, ::MIME"text/plain", r::TuningResult)
    ref = findfirst(==(1), r.packs)
    t0 = ref === nothing ? minimum(r.times) : r.times[ref]
    label = ref === nothing ? "vs best" : "vs P=1"
    println(io, "TuningResult for ", r.kernel, " — best P = ", r.best)
    println(io, rpad("P", 6), rpad("time", 14), label)
    println(io, "-"^32)
    for (P, t) in zip(r.packs, r.times)
        mark = P == r.best ? "  <-" : ""
        println(io, rpad(P, 6),
                rpad(string(round(t * 1e3; digits = 3), " ms"), 14),
                rpad(string(round(t0 / t; digits = 2), "x"), 8), mark)
    end
end

Base.show(io::IO, r::TuningResult) =
    print(io, "TuningResult(best = ", r.best, ", packs = ", r.packs, ")")

_tune_time(f::F, arrays::NTuple{NA,Any}, ::Nothing) where {F,NA} = begin
    t0 = time_ns()
    apply!(f, arrays...)
    (time_ns() - t0) / 1e9
end

_tune_time(f::F, arrays::NTuple{NA,Any}, scratch) where {F,NA} = begin
    t0 = time_ns()
    apply!(f, arrays...; scratch = scratch)
    (time_ns() - t0) / 1e9
end

"""
    tune(f, make; packs = (1, 2, 4, 8, 16, 32), rounds = 5, scratch = true) -> TuningResult

Measure kernel `f` at several packet sizes and report which one wins on **this** machine.

`make(P)` must build and return the tuple of arrays for packet size `P`, in the order `f`
expects them. It is called again before every timed round, so in-place recurrences start each
measurement from the same state and the rebuild stays outside the timed region.

`P` is a layout parameter, not the hardware SIMD width: on a 128-bit NEON machine `P = 16`
routinely beats `P = 4`, because a recurrence is latency-bound and several vectors in flight
hide the dependency chain. It has to be measured, which is what this function is for.

- `packs`: packet sizes to try. Each must be a power of two.
- `rounds`: timed repetitions per size; the minimum is kept.
- `scratch`: `true` allocates one instance of workspace with [`scratchlike`](@ref), `false`
  calls the kernel without one, and anything else is passed through as the prototype.

!!! note "A decision tool, not a publication measurement"
    `tune` uses a plain minimum over a few rounds. It reliably separates variants that differ
    by tens of percent or more, which is what choosing `P` requires. For a number you intend
    to publish, use BenchmarkTools and corroborate the vectorization structurally, as
    `bench/` does.

# Example
```julia
result = tune(thomas!, P -> setup_arrays(65_536, 64, P))
result.best        # 16
result[8]          # wall time at P = 8, in seconds
```
"""
function tune(f::F, make::G;
              packs = (1, 2, 4, 8, 16, 32),
              rounds::Int = 5,
              scratch = true) where {F,G}
    rounds >= 1 || throw(ArgumentError("rounds must be at least 1, got $rounds"))
    isempty(packs) && throw(ArgumentError("`packs` must list at least one packet size"))
    for P in packs
        (P isa Integer && P >= 1 && ispow2(P)) ||
            throw(ArgumentError("packet size must be a power of two, got P=$P"))
    end

    sizes = Int[]
    times = Float64[]
    for P in packs
        arrays = _tune_arrays(make, P)
        sc = _tune_scratch(scratch, first(arrays))
        _tune_time(f, arrays, sc)                    # échauffement : compilation et pages
        best = Inf
        for _ in 1:rounds
            arrays = _tune_arrays(make, P)           # état neuf, hors de la mesure
            sc = _tune_scratch(scratch, first(arrays))
            best = min(best, _tune_time(f, arrays, sc))
        end
        push!(sizes, Int(P))
        push!(times, best)
    end

    TuningResult(sizes, times, sizes[argmin(times)], _kernel_name(f))
end

function _tune_arrays(make::G, P) where {G}
    arrays = make(P)
    arrays isa Tuple ||
        throw(ArgumentError(
            "`make(P)` must return a tuple of arrays in the order the kernel expects, " *
            "got $(typeof(arrays))"))
    isempty(arrays) && throw(ArgumentError("`make(P)` returned no arrays"))
    arrays
end

@inline _tune_scratch(flag::Bool, A) = flag ? scratchlike(A) : nothing
@inline _tune_scratch(proto, _) = proto
