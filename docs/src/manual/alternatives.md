# Positioning and alternatives

Interleave is neither a new numerical algorithm nor a replacement for Julia's compiler. It is
a narrow programming model for one awkward case: many independent copies of a loop-carried
recurrence on a CPU.

## The decision table

| approach | best at | recurrence inside one instance | source impact | relationship to Interleave |
|---|---|---|---|---|
| Julia/LLVM auto-vectorization | regular contiguous loops | usually blocked | none | always try first |
| `@simd` | loops known to have independent iterations | illegal for a true recurrence | one annotation and a correctness promise | alternative only on another legal axis |
| LoopVectorization | independent loop nests, reductions, stencils | assumes iteration independence | macro around the loop | preferable for depthwise/Sobel-like kernels |
| Tullio | tensor contractions, convolutions, stencils | not a general feedback-recurrence language | rewrite in index notation | domain-oriented alternative for regular array algebra |
| explicit SIMD.jl | complete manual control | works if lanes are independent instances | vector loads, stores, tails, and packet-aware code | low-level foundation used by Interleave |
| task parallelism | coarse independent jobs | yes, one recurrence per task | parallel driver/scheduler | complementary; Interleave keeps it explicit |
| domain/vendor library | a standard primitive it already implements | library-specific | adapt data and API | often the production choice when it matches |
| GPU kernel/library | very large device-resident parallel workloads | map instances to threads or change algorithm | device code/data management | different scale and hardware target |

!!! note "A second axis, measured"
    The table above sorts by *what the tool is good at*. A second question sorts the outcome
    just as strongly: **how many predecessors the recurrence carries**. A hand-written SoA
    layout beats DLI on a first-order recurrence and loses by 8× on a sixteenth-order one, and
    a kernel with no recurrence at all is where DLI loses outright. See
    [What you would write instead](what-it-replaces.md) for the sweeps.

## What the compiler can and cannot infer

Julia often lets LLVM vectorize straightforward loops automatically. The official
[performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/) warn that
`@simd` promises reorderable iterations and may produce wrong results when dependencies are
present. A Thomas forward sweep or IIR feedback loop has exactly such a dependency.

Interleave does not ask the compiler to disprove it. It changes the element from `Float32` to
`Vec{P,Float32}` so each ordinary arithmetic expression already denotes `P` independent
operations. The sequential loop remains sequential.

## Loop transformation tools

[`LoopVectorization.@turbo`](https://juliasimd.github.io/LoopVectorization.jl/stable/api/)
models nested loops, selects an order, and emits vectorized code. Its documented limitations
include the assumption that loop iterations are independent. That makes it a natural tool
for stencils, convolutions, and many reductions, but not a legal annotation for the recurrence
axis targeted by Interleave.

[`Tullio.jl`](https://github.com/mcabbott/Tullio.jl) expresses convolutions, stencils,
broadcasts, and reductions in index notation and can cooperate with loop vectorization and
threading. It gives up the “keep an arbitrary scalar imperative kernel” goal in exchange for
a much richer optimiser view of regular tensor algebra.

These tools are not universally competing. A real application can use Tullio or
LoopVectorization for an explicit stencil stage and Interleave for the following implicit solve.

## Explicit vector programming

[`SIMD.jl`](https://github.com/eschnett/SIMD.jl) exposes vector types and operations directly.
It makes legality explicit and is the foundation of `Vec{P,T}` in Interleave. Used alone, it also
requires the programmer to arrange loads, stores, tails, and the mapping between lanes and
instances. That is appropriate when every instruction matters or when the operation cannot
be expressed generically.

Interleave contributes the missing policy layer: a logical scalar batch, dense packet storage,
instance views, padding rules, scratch ownership, and sequential or explicitly parallel
drivers. The kernel remains ordinary Julia source.

[`StructArrays.jl`](https://juliaarrays.github.io/StructArrays.jl/stable/) solves a different
layout problem: it presents an array of structures while storing one array per field. It can
be complementary for structured model parameters, but it does not select a SIMD batch axis or
drive recurrence kernels.

## Threads and GPUs

Low-overhead task schedulers such as [OhMyThreads](https://juliafolds2.github.io/OhMyThreads.jl/stable/refs/api/)
or [`Polyester.@batch`](https://github.com/JuliaSIMD/Polyester.jl) distribute independent jobs
across CPU cores. Interleave deliberately separates this choice: `apply!` never launches tasks,
and `parallel_apply!` makes scheduling visible. This avoids hidden nested parallel regions and
lets SIMD and thread scaling be measured independently.

[`CUDA.jl`](https://cuda.juliagpu.org/stable/) offers both array programming and custom Julia
kernels on NVIDIA GPUs. Vendor libraries can go further for standard operations; for example,
[cuSPARSE](https://docs.nvidia.com/cuda/cusparse/#batched-tridiagonal-solve) implements batched
tridiagonal solvers, including an interleaved layout. GPUs are strong when the population is
large and already device-resident. Interleave instead targets composable CPU kernels, small or
medium working sets, and applications where changing numerical source is undesirable.

## Domain libraries

Use a specialised library when the workload is already one of its primitives:

- `LinearAlgebra.Tridiagonal` for idiomatic tridiagonal matrices and specialised solves;
- [DSP.jl](https://docs.juliadsp.org/stable/filters/) for filter design, filter forms, and
  stateful signal processing;
- [NNlib](https://fluxml.ai/NNlib.jl/stable/) for depthwise convolution, differentiation rules,
  and CPU/GPU neural-network integration;
- [MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/dev/) for symbolic PDE discretisation
  and SciML solver composition.

The value of Interleave rises when the kernel is custom, recurrence-heavy, and embedded in a
larger CPU workflow. It falls when a mature library already implements the exact operation.

## What is genuinely distinctive

Data interleaving and cross-instance SIMD are established ideas. The distinctive part of
Interleave.jl is their Julia interface and its constraint: the scalar kernel should remain the
only kernel. A normal `Base.Array` is the scalar configuration; a `Interleave.Array` changes the
element type presented to the same code; `P=1` cleanly returns control to ordinary compiler
vectorization.

That constraint produces a useful form of performance portability across algorithms, not a
promise that every kernel gets faster. The two negative benchmark cases are essential evidence
for that boundary.
