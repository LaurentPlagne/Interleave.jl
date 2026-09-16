# Interleave.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://laurentplagne.github.io/Interleave.jl/dev/)
[![CI](https://github.com/LaurentPlagne/Interleave.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/LaurentPlagne/Interleave.jl/actions/workflows/ci.yml)
![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)

**Keep the recurrence. Vectorize the population.**

Interleave applies one sequential kernel to a batch of independent problems using Data Layout
Interleaving. A change of array type turns scalar elements into SIMD packets; the kernel
source remains unchanged.

```julia
const Arr = Base.Array{Float32,2}        # development and scalar oracle
const Arr = Interleave.Array{Float32,2,8}   # SIMD across eight independent jobs
```

It is designed for recurrence-heavy workloads such as tridiagonal solves, feedback filters,
time-stepping ensembles, and implicit grid-line sweeps. It is usually not useful when LLVM
already vectorizes the ordinary contiguous loop well; in that case, use a standard array or
choose `P=1`.

- `apply!` traverses packets sequentially and never launches tasks.
- `parallel_apply!` adds explicit task parallelism.
- `gpu_apply!` maps one sequential problem to each GPU work item through
  KernelAbstractions; the same scalar Julia kernels used by the tests run on Metal, CUDA,
  and AMDGPU when their vendor runtimes are available.
- Results are tested for exact scalar-to-packed agreement.
- Packet size is a tuning parameter to measure, not a performance guarantee.

Start with the visual [documentation and tutorials](https://laurentplagne.github.io/Interleave.jl/dev/), especially the
[tridiagonal recurrence tutorial](https://laurentplagne.github.io/Interleave.jl/dev/tutorials/thomas/). The
[benchmark application studies](https://laurentplagne.github.io/Interleave.jl/dev/applications/) cover all shipped
kernels with animations, complete performance tables, and comparisons with alternative
compiler, library, threaded, and GPU approaches.

The experimental [GPU design and tutorial](https://laurentplagne.github.io/Interleave.jl/dev/manual/gpu/) explains the
KernelAbstractions driver, batch-major device layout, and the shared all-kernel suite.

## Reproducible target benchmarks

The GitHub Actions workflow `Cross-platform benchmarks` runs on pushes to `main`, can be
launched manually, on a release, or weekly. It runs the CPU suite on Linux and Apple Silicon,
and benchmarks every validation kernel on Metal macOS.

NVIDIA and AMD numbers are **not** produced by CI. GitHub's managed GPU runners require an
organization on a Team or Enterprise plan, which this repository is not, so the `ka-nvidia`
and `ka-amd` jobs stay dormant unless `INTERLEAVE_NVIDIA_RUNNER` / `INTERLEAVE_AMD_RUNNER`
name a self-hosted runner. To produce NVIDIA numbers on any machine — a workstation, a rented
box, or a Colab runtime — use `./gpu/run_remote.sh cuda`; only the vendor driver is required,
since CUDA.jl ships its own toolkit. See the [GPU manual](https://laurentplagne.github.io/Interleave.jl/dev/manual/gpu/) for the
three available paths. The Vulkan/SPIR-V prototype is deliberately outside this workflow.

[When *not* to use Interleave](https://laurentplagne.github.io/Interleave.jl/dev/manual/what-it-replaces/) compares every shipped kernel
against what you would write in plain Julia or with another package — including the two cases
where Interleave is slower than doing nothing.

Interleave.jl is experimental. It is not registered yet; install the repository directly:

```julia
pkg> add https://github.com/laurentplagne/Interleave.jl
```
