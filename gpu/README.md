# Experimental GPU backends

Interleave maps naturally to a GPU, but not by moving `Vec{P,T}` to the device. One GPU
work item owns one independent problem and executes its recurrence sequentially. The
work items of a SIMT group collectively replace the CPU vector lanes.

All device buffers use the Julia shape `(nbatch, dims...)`. The first dimension is
contiguous, so neighbouring work items access neighbouring scalars at every recurrence
step.

The maintained GPU path is one KernelAbstractions driver with vendor-specific storage:

- [`ka/all.jl`](ka/all.jl) is the backend-neutral benchmark entry point. Set
  `INTERLEAVE_KA_BACKEND=metal`, `cuda`, or `amdgpu`.
- [`metal/`](metal/) contains the Apple Silicon environment and the shared implementation.
- [`cuda/`](cuda/) and [`amd/`](amd/) contain the CUDA and ROCm environments used by the
  optional GitHub runner jobs.

The CPU package never loads KernelAbstractions.jl or a vendor runtime. The generic GPU driver
is a weak-dependency extension activated by KernelAbstractions. The old [`vulkan/`](vulkan/)
directory is retained as an explicitly unsupported ABI experiment, but is no longer built or
benchmarked by CI.

GitHub's managed GPU larger runners are **not** available to this repository: they require an
organization on a Team or Enterprise plan, and this is a personal account. The `ka-nvidia` and
`ka-amd` jobs therefore expect the label of a **self-hosted** runner in
`INTERLEAVE_NVIDIA_RUNNER` / `INTERLEAVE_AMD_RUNNER`, and stay dormant without one.

For a one-shot measurement on any GPU machine, use [`run_remote.sh`](run_remote.sh):

```bash
./gpu/run_remote.sh cuda     # or amdgpu, or metal; omit to guess from the hardware
```

It installs Julia if needed, records the device, resolves the matching environment, and runs
the suite. Only the vendor driver has to be present — CUDA.jl and AMDGPU.jl ship their own
toolchains as Julia artifacts.
