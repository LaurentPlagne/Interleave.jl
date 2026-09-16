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
- an optional NVIDIA CUDA job, dormant until a runner label is configured;
- an optional AMDGPU/ROCm job, dormant on the same terms.

Each job uploads its raw output and appends it to the GitHub job summary.

### Obtaining NVIDIA numbers

!!! warning "GitHub's managed GPU runners are not available here"
    GitHub-hosted *larger* runners, GPU ones included, require an organization on a Team or
    Enterprise Cloud plan. `LaurentPlagne/Interleave.jl` is a personal account, so there is
    no runner to create and no label to copy. Earlier revisions of this page and of the
    workflow described a Tesla-T4 setup that cannot be performed on this repository.

Three paths actually work, in increasing order of commitment:

1. **Run the suite by hand on any NVIDIA machine.** This is what Legolas++ did: its published
   NVIDIA figures come from a workstation with a GeForce RTX 2060 SUPER, not from CI.
   [`gpu/run_remote.sh`](https://github.com/laurentplagne/Interleave.jl/blob/main/gpu/run_remote.sh)
   installs Julia if needed, records the device, resolves `gpu/cuda`, and runs the suite:

   ```bash
   git clone https://github.com/LaurentPlagne/Interleave.jl && cd Interleave.jl
   ./gpu/run_remote.sh cuda
   ```

   It works unchanged on a rented box (RunPod, Vast.ai, Lambda), on a Colab runtime, and on
   an institutional workstation. Only the vendor *driver* is required: CUDA.jl ships its own
   toolkit as Julia artifacts.

2. **Register a self-hosted runner** under *Settings → Actions → Runners*, then set the
   repository variable `INTERLEAVE_NVIDIA_RUNNER` (or the `nvidia_runner` dispatch input) to
   its label. The `ka-nvidia` job then runs on releases and on a manual dispatch with
   `run_gpu=true`. A self-hosted runner on a public repository is only safe because this
   workflow has no `pull_request` trigger; do not add one.

3. **Use the JuliaGPU Buildkite infrastructure**, which is how CUDA.jl and
   KernelAbstractions.jl test. It is free and permanent, but requires coordination on the
   JuliaLang Slack `#gpu` channel. See [JuliaGPU/buildkite](https://github.com/JuliaGPU/buildkite).

The same three options apply to AMDGPU through `INTERLEAVE_AMD_RUNNER`; GitHub publishes no
AMD/ROCm hosted label at all.

## Measured results

The same ten scalar kernels, the same driver, the same commit, on two backends. Kernel time
only — no host transfers — and every case is validated against the scalar CPU oracle before it
is timed.

| kernel | recurrence? | Metal, M1 Max | CUDA, T4 |
|---|:---:|---:|---:|
| Thomas | yes | 0.586 ms | **0.245 ms** |
| Biquad | yes | 0.555 ms | **0.164 ms** |
| Black–Scholes CN | yes | 1.800 ms | **0.734 ms** |
| Thomas lines | yes | 0.632 ms | **0.202 ms** |
| Tridiagonal product | yes | 0.333 ms | **0.056 ms** |
| Squared norm | reduction | 0.446 ms | **0.131 ms** |
| Dot product | reduction | 0.514 ms | **0.162 ms** |
| Depthwise 3×3 | **no** | **2.854 ms** | 5.487 ms |
| Sobel + motion | **no** | **3.169 ms** | 3.816 ms |
| Laplacian 3D | **no** | **4.435 ms** | 5.850 ms |

The split is clean and follows the same line as everywhere else in this documentation. On the
recurrence kernels and the reductions the T4 wins by 2.4× to 5.9×. On the three **stencils** it
*loses*, by 1.2× to 1.9×.

That is not a statement about the hardware. It is the known limitation of this driver: one work
item owns one complete instance, so a stencil kernel becomes a single thread looping over a
whole image. That proves source reuse, which is the point of the exercise, but it is not a
performance-optimal stencil schedule — a real one decomposes over pixels or tiles. The T4's
weaker per-thread execution exposes that more than the M1 Max's does.

!!! warning "Not a hardware comparison"
    Different machines, different memory systems, and the T4 was a shared Colab instance.
    Kernel-only timings exclude transfers, which dominate for a discrete GPU unless the data is
    already resident. Read the table as evidence that one source runs on both and that the
    *shape* of the result follows the kernel type — not as a benchmark of Apple against NVIDIA.

### Bit-exactness stops at the backend

This is the finding worth carrying away, and it qualifies the package's central invariant.

| | Metal | CUDA |
|---|---|---|
| kernels bit-exact against the CPU oracle | **10 / 10** | **1 / 10** |

Metal reproduces the scalar CPU result *exactly* — `max error 0.0` on every kernel. CUDA does
not: errors from `1.2e-7` on the biquad to `7.3e-4` on Black–Scholes, which is a doubly nested
recurrence and accumulates the difference over eight time steps.

The cause is almost certainly **FMA contraction**: NVIDIA's compiler fuses `a*b + c` into a
single `fma` by default, which rounds once instead of twice. That is precisely the
transformation [invariant 1](https://github.com/LaurentPlagne/Interleave.jl/blob/main/AGENTS.md)
forbids on the CPU path, where `@fastmath` is banned for exactly this reason — and on the GPU
it is the vendor compiler's default, outside this package's control.

The practical consequence:

- the **CPU** guarantee is unchanged. `Vec{P,T}` results are bit-identical (`==`, not `≈`) to
  the scalar kernel, and the test suite asserts it;
- the **GPU** path guarantees *the same algorithm*, not the same rounding, and how close the
  results are is a property of the backend. The validation suite therefore compares with a
  tolerance rather than equality, and that tolerance is not a weakness in the test — it is the
  honest statement of what a vendor compiler leaves you.

If you need bit-exact agreement with the CPU reference, Metal currently provides it and CUDA
does not. Do not assume it; the suite prints the measured error for every kernel.

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
