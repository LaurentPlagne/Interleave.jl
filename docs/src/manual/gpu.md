# GPU execution: recurrence per work item

The GPU translation keeps the mathematical idea and changes the machine mapping:

| | CPU DLI | GPU DLI |
|---|---|---|
| sequential unit | one `Vec{P,T}` recurrence | one scalar recurrence per work item |
| parallel population | SIMD lanes | a SIMD-group / wave of work items |
| storage | `Array{Vec{P,T}}` | scalar `(nbatch, dims...)` device array |
| tuning knob | packet size `P` | workgroup size and resident batch size |

At recurrence position `i`, GPU work items `b`, `b+1`, … read
`A[b, i]`, `A[b+1, i]`, … . Julia stores the first dimension contiguously, so the
population is coalesced without a packed element type. The recurrence is still
sequential *inside* each work item.

This is the GPU analogue of the Interleave idea, but it is SIMT rather than explicit CPU
SIMD. A device backend may still split or schedule a workgroup in hardware-specific
ways; `gpu_apply!` does not promise a particular machine instruction width.

## Metal tutorial

Install Metal.jl in the application environment, then keep each batch as an ordinary
scalar matrix before uploading it:

```julia
using Interleave, Metal

X, D, U, L, B = make_thomas_batch(Float32, nbatch, nx)
dX, dD, dU, dL, dB = MtlArray.((X, D, U, L, B))
dS = similar(dX)                    # one scratch row per independent problem

gpu_apply!(thomas!, dX, dD, dU, dL, dB;
           scratch=dS, workgroupsize=256)

# Do more GPU work here; the launch above is asynchronous.
gpu_synchronize(dX)
result = Array(dX)
```

The `thomas!` function can be the same scalar Julia function used by [`apply!`](@ref).
The driver creates a small, allocation-free instance object inside every work item and
passes those objects to the function. See the complete runnable
[`gpu/metal/thomas.jl`](https://github.com/laurentplagne/Interleave.jl/blob/main/gpu/metal/thomas.jl)
prototype.

There is one important qualification to “the same kernel”: Metal's compiler accepts a
restricted, statically dispatched subset of Julia. Device code cannot allocate, throw,
perform I/O, create tasks, or call dynamically selected methods. Scalar CPU success is
necessary, not sufficient, for GPU compilation.

## What happened to `P`?

There is no `P` on the GPU path. `Vec{P,T}` is the CPU mechanism that forces several
independent scalar values through one LLVM vector operation. On a GPU, separate work
items already execute in lockstep. Nesting a `Vec` inside every work item would add a
second level of vectorization, increase register pressure, and usually reduce occupancy.

Tune `workgroupsize` instead. Start at 128 or 256, benchmark neighbouring values, and
include transfer costs only when the application really transfers for every call.

## Vulkan and SPIR-V

The Vulkan prototype deliberately stops at a stable shader ABI:

- six `std430` scalar storage buffers for `X`, `D`, `U`, `L`, `B`, and scratch `S`;
- offset `problem + i*stride`, identical to a Julia matrix `(stride, nx)`;
- three `UInt32` push constants: `nbatch`, `nx`, and `stride`;
- specialization constant 0 for the workgroup width;
- one invocation per tridiagonal system.

[`gpu/vulkan/shaders/thomas.comp`](https://github.com/laurentplagne/Interleave.jl/blob/main/gpu/vulkan/shaders/thomas.comp)
is compiled and validated as Vulkan 1.2 SPIR-V by the accompanying script. Vulkan.jl
can build a compute pipeline from that module.

This is not yet a second implementation of [`gpu_apply!`](@ref). Vulkan.jl is a
low-level Vulkan wrapper, not a Julia GPU compiler. The former JuliaGPU `SPIRV.jl`
compiler is archived, while MLIR.jl does not currently provide a production Julia-to-
Vulkan kernel path. Hiding hand-written GLSL behind the same function name would imply
source portability that the stack cannot deliver.

The next Vulkan milestone is therefore host-side infrastructure rather than another
shader:

1. retain a Vulkan instance, compute queue, command pool, and descriptor pool;
2. allocate a reusable GPU-local buffer arena and staging buffers;
3. cache pipelines by shader and workgroup specialization;
4. submit asynchronously and make object lifetimes explicit;
5. compare the same frozen scalar oracle used by the CPU and Metal tests.

Only after that runtime is measured should a Julia-to-SPIR-V or MLIR lowering layer be
considered. It is a compiler project, not a container change.

## Correctness contract

The GPU tests must preserve the properties that matter:

- no work item reads or writes another problem;
- a non-multiple of the workgroup width is masked correctly;
- sequential, Metal, and Vulkan results are compared against the same scalar oracle;
- repeated asynchronous launches are deterministic;
- no hidden host/device copy occurs in the timed region.

Cross-device *bit identity* should be tested but not advertised before it is observed on
every supported backend. Different GPU division implementations and contraction rules
may prevent the stronger CPU SIMD guarantee even when the source operation order is
unchanged.

## When a GPU is the wrong target

GPU launch and transfer overhead can dominate short recurrences or small batches. The
GPU path is attractive when the population is large, the data remains resident across
several operations, and each problem exposes enough sequential work to amortize launch
cost without exhausting registers. CPU DLI remains the low-latency path.

