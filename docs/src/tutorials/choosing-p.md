# Tutorial 4 — Decide first, then tune `P`

Interleave is a targeted optimization. This tutorial prevents the most common mistake: packing
a kernel that LLVM already vectorizes well.

## 1. Measure the ordinary kernel

Start from the `Base.Array` path and measure behind a function barrier. Record both runtime
and a meaningful throughput. A low throughput is often evidence of a dependency chain; a
high throughput suggests the compiler already found SIMD along a contiguous axis.

![Measured reference throughput predicts DLI benefit](../assets/fit-map.svg)

The separation in this dataset is striking, but it is not a universal numerical threshold.
Use it as a question generator: inspect LLVM and identify the dependency that explains the
measurement.

## 2. Sweep powers of two

Valid packet sizes are powers of two. Benchmark at least `P ∈ (1, 2, 4, 8, 16, 32)` when
the element type and target allow it.

```julia
function run_case(::Val{P}, setup, kernel!) where P
    arrays = setup(Interleave.Array{Float32,2,P})
    apply!(kernel!, arrays...)
    arrays
end
```

Keep each timed case in its own function. Alternate variants rather than timing all runs of
one variant and then all runs of another; this reduces bias from changing machine load.

## 3. Understand why `P` can exceed hardware width

On a 128-bit NEON unit, one native vector holds four `Float32` values. Yet `P=32` was best
for the measured sequential Thomas solve:

![Thomas speedup for packet sizes one through thirty-two](../assets/thomas-speedup.svg)

A recurrence is latency-bound. A larger packet type can compile into several hardware
vectors, giving the processor independent instructions to keep in flight while earlier
operations wait. `P` is therefore an unrolling and latency-hiding parameter as well as a
lane count.

The optimum can move when threads are enabled: the same Thomas problem preferred `P=16`
with ten Julia threads. At `P=64`, native-code inspection showed vector stack traffic and no
further sequential gain. Sweep wide enough to observe both improvement and its plateau;
do not assume the largest supported packet wins.

## 4. Treat `P=1` as a real outcome

`packtype(T, Val(1))` is exactly `T`, not `Vec{1,T}`. This matters: a one-lane vector can
block LLVM from auto-vectorizing an otherwise ordinary contiguous loop.

For a stencil, convolution, or map-like kernel with no recurrence, the correct result of
tuning may be `P=1`. That is not a failure. It means the standard compiler strategy already
matches the problem.

## 5. Add threads only after SIMD tuning

Once the best sequential `P` is known, measure [`parallel_apply!`](@ref). SIMD and threads
do not necessarily multiply:

- a compute-bound repeated solve may scale well across cores;
- a streaming solve that reads several arrays may hit memory bandwidth early;
- task-launch overhead dominates small batches.

Always compare these three configurations:

1. scalar reference;
2. best sequential `apply!` configuration;
3. the same `P` with `parallel_apply!`.

If the third result is not optimal, repeat the `P` sweep with threading enabled. The
sequential optimum is only a starting point because memory bandwidth and register pressure
change when several cores execute packets concurrently.

## 6. Automate the protocol, not the decision context

An opt-in tuner can construct all packet layouts, check each one against the scalar oracle,
inspect LLVM, benchmark through BenchmarkTools, and return the fastest valid candidate.
What it cannot infer is whether setup and packing belong in the timed operation, which
problem sizes are representative, or whether exact equality is the appropriate numerical
contract.

Cache a tuned choice only with its context: kernel type, scalar type, shape class, thread
mode, Julia/LLVM version, and CPU. Because `P` determines the array's concrete storage type,
selection must happen before production arrays are allocated; it should not be an invisible
branch inside `apply!`.

## 7. Use a stop rule

Keep DLI only when all of the following are true:

- the end-to-end workload improves, including layout conversion;
- LLVM contains the expected vector types;
- packed results exactly match the scalar oracle where exactness is required;
- the best configuration remains stable across representative sizes;
- the added memory footprint and padding are acceptable.

If any condition fails, retain the same kernel and select `P=1` or a normal array.
