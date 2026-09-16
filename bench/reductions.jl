# Per-instance reductions from the VulkanBench example.
# The same scalar kernel is used by apply!, packed CPU arrays, and gpu_apply!;
# only the storage and execution backend change.
include(joinpath(@__DIR__, "harness.jl"))
include(joinpath(@__DIR__, "..", "test", "kernels.jl"))
const T = Float32

function reduction_inputs(nbatch, n)
    A = [sinpi(T(b) / 17) + T(i) / n for b in 1:nbatch, i in 1:n]
    B = [cospi(T(b) / 23) - T(i) / (2n) for b in 1:nbatch, i in 1:n]
    A, B
end

function reduction_case(name, f, arrays, flops; rounds)
    reference = zeros(T, size(first(arrays), 1), 1)
    apply!(f, reference, arrays...)
    variants = Pair{String,Any}["référence" => () -> apply!(f, reference, arrays...)]
    resets = Function[() -> fill!(reference, zero(T))]

    sets = map(P -> begin
        packed = map(A -> Interleave.Array(A; pack = Val(P)), arrays)
        output = Interleave.Array{T}(undef, size(first(arrays), 1), 1; pack = Val(P))
        fill!(output, zero(T))
        (output, packed...)
    end, PACKS)

    for (P, set) in zip(PACKS, sets)
        output, packed = set[1], Base.tail(set)
        push!(variants, "P=$P" => let f = f, output = output, packed = packed
            () -> apply!(f, output, packed...)
        end)
        initial = deepcopy(output)
        push!(resets, let output = output, initial = initial
            () -> copyto!(output, initial)
        end)
        push!(variants, "P=$P threadé" => let f = f, output = output, packed = packed
            () -> parallel_apply!(f, output, packed...; scheduler = StaticScheduler())
        end)
        push!(resets, let output = output, initial = initial
            () -> copyto!(output, initial)
        end)
    end

    best = interleaved(variants; rounds, resets)
    table(best, ["référence"; ["P=$P" for P in PACKS]], flops, "référence")
    println()
    table(best, ["référence"; ["P=$P threadé" for P in PACKS]], flops, "référence")
    best
end

function run(; nbatch = 16_384, n = 256, rounds = 5)
    A, B = reduction_inputs(nbatch, n)
    header("Per-instance reductions (same kernel on CPU and GPU)",
           "$nbatch independent vectors × $n values — reduction inside each instance",
           "values", nbatch * n)
    println("\nSquared norm")
    reduction_case("squared norm", batch_squarednorm!, (A,), 2 * nbatch * n; rounds)
    println("\nDot product")
    reduction_case("dot product", batch_dot!, (A, B), 3 * nbatch * n; rounds)
end
