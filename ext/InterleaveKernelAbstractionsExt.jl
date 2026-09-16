module InterleaveKernelAbstractionsExt

using Interleave
import Interleave: gpu_apply!, gpu_backend, gpu_synchronize, gpu_scratchlike
import KernelAbstractions
using KernelAbstractions: @index, @kernel

# One allocation-free view is constructed inside each work item. The parent device array
# keeps the batch as its first and contiguous dimension.
struct DeviceInstance{T,N,A} <: AbstractArray{T,N}
    parent::A
    batch::Int
end

@inline DeviceInstance(A::AbstractArray{T,M}, batch::Int) where {T,M} =
    DeviceInstance{T,M - 1,typeof(A)}(A, batch)

Base.size(A::DeviceInstance) = Base.tail(size(A.parent))
Base.IndexStyle(::Type{<:DeviceInstance}) = IndexCartesian()

@inline function Base.getindex(A::DeviceInstance{T,N}, I::Vararg{Int,N}) where {T,N}
    A.parent[A.batch, I...]
end

@inline function Base.setindex!(A::DeviceInstance{T,N}, value,
                                I::Vararg{Int,N}) where {T,N}
    A.parent[A.batch, I...] = value
end

@inline function _device_apply(f::F, arrays::NTuple{NA,Any}, ::Nothing,
                               batch::Int) where {F,NA}
    f(ntuple(i -> DeviceInstance(arrays[i], batch), Val(NA))...)
    nothing
end

@inline function _device_apply(f::F, arrays::NTuple{NA,Any}, scratch,
                               batch::Int) where {F,NA}
    f(ntuple(i -> DeviceInstance(arrays[i], batch), Val(NA))...,
      DeviceInstance(scratch, batch))
    nothing
end

@kernel function _gpu_apply_kernel!(f, arrays, scratch, nbatch)
    batch = @index(Global, Linear)
    if batch <= nbatch
        _device_apply(f, arrays, scratch, batch)
    end
    nothing
end

@inline gpu_backend(A::AbstractArray) = KernelAbstractions.get_backend(A)

# Le pendant GPU de `scratchlike`. Volontairement un **autre nom** plutôt qu'une méthode de
# `scratchlike` : les deux objets ne sont pas interchangeables, et un nom distinct empêche de
# passer l'un là où l'autre est attendu sans s'en apercevoir. Ici le workspace est un lot
# batch-major complet, dont chaque work item ne voit que sa propre tranche.
gpu_scratchlike(A::AbstractArray) = similar(A)

function gpu_synchronize(A::AbstractArray)
    KernelAbstractions.synchronize(gpu_backend(A))
    nothing
end

function _checked_gpu_batch(arrays::NTuple{NA,Any}, scratch) where NA
    isempty(arrays) && throw(ArgumentError("gpu_apply! requires at least one array"))
    A = first(arrays)
    ndims(A) >= 2 || throw(ArgumentError(
        "a GPU batch needs a batch axis and at least one instance axis"))
    nbatch = size(A, 1)
    backend = gpu_backend(A)
    for B in arrays
        ndims(B) >= 2 || throw(ArgumentError(
            "a GPU batch needs a batch axis and at least one instance axis"))
        size(B, 1) == nbatch || throw(DimensionMismatch(
            "batch sizes differ: $nbatch and $(size(B,1))"))
        gpu_backend(B) == backend || throw(ArgumentError(
            "all arrays passed to gpu_apply! must use the same backend"))
    end
    if scratch !== nothing
        ndims(scratch) >= 2 || throw(ArgumentError(
            "GPU scratch must have a batch axis and at least one instance axis"))
        size(scratch, 1) == nbatch || throw(DimensionMismatch(
            "scratch batch size $(size(scratch,1)) differs from $nbatch"))
        gpu_backend(scratch) == backend || throw(ArgumentError(
            "GPU scratch must use the same backend as the input arrays"))
    end
    (; backend, nbatch)
end

function gpu_apply!(f::F, arrays::Vararg{AbstractArray,NA};
                    scratch = nothing,
                    workgroupsize::Int = 256,
                    wait::Bool = false) where {F,NA}
    workgroupsize > 0 || throw(ArgumentError("workgroupsize must be positive"))
    backend, nbatch = _checked_gpu_batch(arrays, scratch)
    kernel! = _gpu_apply_kernel!(backend, workgroupsize)
    kernel!(f, arrays, scratch, nbatch; ndrange = nbatch)
    wait && KernelAbstractions.synchronize(backend)
    first(arrays)
end

end # module
