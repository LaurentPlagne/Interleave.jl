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

The CUDA job targets the name/label assigned when a GitHub-managed Tesla-T4 larger runner is
created, on releases or an explicit GPU workflow dispatch. Set `INTERLEAVE_NVIDIA_RUNNER` (or
the dispatch input) to that name. There is no standard GitHub-hosted AMD/ROCm label; set
`INTERLEAVE_AMD_RUNNER` to the exact label of an AMD-provided or institutional ROCm runner.
