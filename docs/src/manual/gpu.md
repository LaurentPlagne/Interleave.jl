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

## KernelAbstractions tutorial

Install the vendor package in the application environment, then keep each batch as an
ordinary scalar matrix before uploading it. The Julia kernel is unchanged across targets:

```julia
using Interleave, Metal                 # or CUDA / AMDGPU

X, D, U, L, B = make_thomas_batch(Float32, nbatch, nx)
dX, dD, dU, dL, dB = MtlArray.((X, D, U, L, B)) # CuArray / ROCArray on other targets
dS = similar(dX)                    # one scratch row per independent problem

gpu_apply!(thomas!, dX, dD, dU, dL, dB;
           scratch=dS, workgroupsize=256)

# Do more GPU work here; the launch above is asynchronous.
gpu_synchronize(dX)
result = Array(dX)
```

The `thomas!` function is the same scalar Julia function used by [`apply!`](@ref). The
driver creates a small, allocation-free instance object inside every work item and passes
those objects to the function. The same contract is exercised for Thomas, biquad,
depthwise convolution, Sobel plus motion, Black–Scholes, a 3-D Laplacian, Thomas line
sweeps, tridiagonal products, and reductions by the runnable
[`gpu/ka/all.jl`](https://github.com/laurentplagne/Interleave.jl/blob/main/gpu/ka/all.jl)
suite. Select a backend with `INTERLEAVE_KA_BACKEND=metal`, `cuda`, or `amdgpu`.

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

## Deferred backends

Vulkan/SPIR-V is intentionally not part of the current API or continuous benchmark. The
repository keeps the old [`gpu/vulkan/`](https://github.com/laurentplagne/Interleave.jl/tree/main/gpu/vulkan)
directory as a historical ABI experiment, but no workflow compiles it and no result from it
is presented as a KernelAbstractions execution. Reintroducing it would require a maintained
Julia-to-SPIR-V compiler and a real host dispatcher, not merely another device-array alias.

## Continuous benchmark targets

The repository workflow mirrors the multi-target structure used by Legolas++:

- Linux x86-64 runs the complete `BenchmarkTools` CPU suite;
- macOS 14 on Apple Silicon runs the same CPU suite and all validation kernels through
  the resident Metal KernelAbstractions driver;
- an optional NVIDIA CUDA job runs on an organization-configured GPU runner;
- an optional AMDGPU/ROCm job runs on an organization-configured self-hosted runner.

The CPU, Metal, CUDA, and AMDGPU jobs upload their raw output and append it to the GitHub job
summary. The CUDA and AMDGPU jobs are skipped until the repository variables
`INTERLEAVE_NVIDIA_RUNNER` and `INTERLEAVE_AMD_RUNNER` are set to real runner names/labels.
This is deliberate: standard GitHub-hosted runners do not provide a GPU, and GitHub does not
provide a universal AMD label.

To enable the jobs, open the repository's **Settings → Secrets and variables → Actions →
Variables** and set `INTERLEAVE_NVIDIA_RUNNER` to the name/label of an organization-configured
GitHub GPU larger runner (normally an NVIDIA T4), and/or set `INTERLEAVE_AMD_RUNNER` to the
name/label of a self-hosted Linux runner with ROCm and AMDGPU.jl installed. The workflow does
not guess labels: a wrong label must fail visibly instead of silently running on a CPU. GitHub's
runner name and label syntax is documented in its [runner selection guide](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job).

## Correctness contract

The GPU tests must preserve the properties that matter:

- no work item reads or writes another problem;
- a non-multiple of the workgroup width is masked correctly;
- sequential and each configured KernelAbstractions backend are compared against the same
  scalar oracle;
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
