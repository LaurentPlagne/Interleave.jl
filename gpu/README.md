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

To activate the CUDA or ROCm jobs, set `INTERLEAVE_NVIDIA_RUNNER` or
`INTERLEAVE_AMD_RUNNER` as repository Actions variables to the exact runner name/label. The
standard GitHub-hosted pool has no GPU; NVIDIA requires an organization-enabled GPU larger
runner, while AMD requires a self-hosted ROCm runner.
