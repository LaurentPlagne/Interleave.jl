# Interleave.jl

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
- `gpu_apply!` maps one sequential problem to each GPU work item; Metal is executable,
  while the Vulkan/SPIR-V path currently fixes and validates the shader ABI.
- Results are tested for exact scalar-to-packed agreement.
- Packet size is a tuning parameter to measure, not a performance guarantee.

Start with the visual [documentation and tutorials](docs/src/index.md), especially the
[tridiagonal recurrence tutorial](docs/src/tutorials/thomas.md). The
[benchmark application studies](docs/src/applications/index.md) cover all five shipped
kernels with animations, complete performance tables, and comparisons with alternative
compiler, library, threaded, and GPU approaches.

The experimental [GPU design and tutorial](docs/src/manual/gpu.md) explain the Metal
driver, batch-major device layout, and the deliberately lower-level Vulkan/SPIR-V
prototype.

Interleave.jl is experimental. It is not registered yet; install the repository directly:

```julia
pkg> add https://github.com/laurentplagne/Interleave.jl
```
