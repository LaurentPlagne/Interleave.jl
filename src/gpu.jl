"""
    gpu_backend(A)

Return the execution backend associated with device array `A`. This method becomes
available when a KernelAbstractions-compatible backend such as Metal.jl is loaded.
"""
function gpu_backend(A)
    throw(ArgumentError(
        "no GPU backend is loaded; load Metal.jl or another KernelAbstractions backend"))
end

"""
    gpu_synchronize(A)

Wait until all previously submitted work on the backend associated with `A` has
completed. [`gpu_apply!`](@ref) is asynchronous unless called with `wait=true`.
"""
function gpu_synchronize(A)
    gpu_backend(A)
    nothing
end

"""
    gpu_scratchlike(A) -> device array

Batch-major workspace for [`gpu_apply!`](@ref): one private row per work item, on the same
backend as `A`. Available once a KernelAbstractions backend is loaded.

This is deliberately **not** the same function as [`scratchlike`](@ref), which returns one
instance of CPU workspace for [`apply!`](@ref). The two shapes are different and are not
interchangeable; separate names make a mix-up impossible to write by accident.
"""
function gpu_scratchlike(A)
    throw(ArgumentError(
        "no GPU backend is loaded; load Metal.jl or another KernelAbstractions backend"))
end

"""
    gpu_apply!(f, arrays...; scratch=nothing, workgroupsize=256, wait=false)

Submit one GPU work item per independent problem. Every array uses the ordinary logical
shape `(nbatch, dims...)`; because Julia is column-major, the batch axis is contiguous
and accesses made by neighbouring GPU work items are coalesced.

The work item receives allocation-free `AbstractArray` views of one problem and calls
`f` with the same signature as [`apply!`](@ref). The recurrence inside `f` remains
sequential. GPU SIMT execution across problems replaces the CPU's `Vec{P,T}` element
type, so there is no packet size `P` on this path.

If `scratch` is needed, pass a *batched device array*, obtained with
[`gpu_scratchlike`](@ref).
Each work item sees only its own slice. Interleave allocates no device or scratch buffer;
the backend may still allocate compiler and launch bookkeeping on the host.

Launches follow normal GPU conventions and are asynchronous. Set `wait=true` for a
blocking call, or call [`gpu_synchronize`](@ref) later. `workgroupsize` must be positive
and should be measured on the target device.

Only the GPU-compilable subset of Julia may appear in `f`: no allocation, exceptions,
I/O, task creation, or runtime dispatch. A kernel that works with [`apply!`](@ref) is
therefore not automatically guaranteed to compile for every GPU backend.

The implementation is an optional package extension. Loading Interleave alone does not load
a GPU compiler. Load Metal.jl or another KernelAbstractions-compatible backend before
calling this function.

# Example
```julia
using Metal

dX, dD, dU, dL, dB = MtlArray.((X, D, U, L, B))
dS = similar(dX)
gpu_apply!(thomas!, dX, dD, dU, dL, dB; scratch=dS, wait=true)
X .= Array(dX)
```
"""
function gpu_apply!(args...; kwargs...)
    throw(ArgumentError(
        "no GPU backend is loaded; load Metal.jl or another KernelAbstractions backend"))
end
