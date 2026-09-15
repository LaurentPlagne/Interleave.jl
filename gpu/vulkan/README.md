# Vulkan/SPIR-V prototype

This directory freezes the portable device ABI before investing in Vulkan's verbose
host-side plumbing.

The compute shader implements Thomas with one invocation per independent system. Its
six storage buffers are scalar `Float32` arrays using
`problem + recurrence_index * stride`, exactly the flat representation of a Julia
matrix shaped `(stride, nx)`. The first three `UInt32` push constants are `nbatch`, `nx`,
and `stride`; specialization constant 0 selects the workgroup width.

[`abi.jl`](abi.jl) defines the matching twelve-byte Julia push-constant structure,
descriptor binding numbers, and checked dispatch geometry. A zero-sized batch produces
zero groups and must be treated as a no-op by the future dispatcher.

Compile and validate the shader by running `compile.jl` in this environment. The
generated `thomas.spv` is intentionally ignored and should be rebuilt for the target
toolchain.

Vulkan.jl can consume the resulting module. The next implementation milestone is a
reusable context owning a compute queue, descriptor pool, pipeline cache, and GPU-local
buffer arena. Those objects must outlive asynchronous submission; building them afresh
for every `gpu_apply!` would make launch overhead dominate small batches.

This path does **not** yet preserve the one-source-kernel promise. The former SPIRV.jl
compiler is archived, and Vulkan.jl currently expects SPIR-V produced from a shading
language. MLIR remains a possible future compiler layer, not a runtime backend available
to this package today.
