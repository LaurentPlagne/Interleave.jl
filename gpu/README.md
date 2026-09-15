# Experimental GPU backends

Interleave maps naturally to a GPU, but not by moving `Vec{P,T}` to the device. One GPU
work item owns one independent problem and executes its recurrence sequentially. The
work items of a SIMT group collectively replace the CPU vector lanes.

All device buffers use the Julia shape `(nbatch, dims...)`. The first dimension is
contiguous, so neighbouring work items access neighbouring scalars at every recurrence
step.

Two deliberately different prototypes live here:

- [`metal/`](metal/) runs the original Julia recurrence through
  `KernelAbstractions.jl` and `Metal.jl`. This is the preferred path on Apple Silicon.
- [`vulkan/`](vulkan/) fixes the same memory layout and runs a GLSL compute shader
  compiled to SPIR-V. It proves the Vulkan ABI, but not source-level reuse: Julia has no
  maintained Julia-to-Vulkan SPIR-V compiler today.

The CPU package never loads KernelAbstractions.jl, Metal.jl, or Vulkan.jl. The generic
GPU driver is a weak-dependency extension activated by KernelAbstractions; Metal users
only need to add Metal.jl to their application environment. Vulkan's extra packages are
confined to the prototype environment.
