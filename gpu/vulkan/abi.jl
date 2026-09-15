"""Fixed descriptor bindings shared by the Vulkan host and Thomas compute shader."""
const THOMAS_BINDINGS = (X = 0, D = 1, U = 2, L = 3, B = 4, S = 5)

"""Twelve-byte push-constant block consumed by `shaders/thomas.comp`."""
struct ThomasPushConstants
    nbatch::UInt32
    nx::UInt32
    stride::UInt32
end

function ThomasPushConstants(nbatch::Integer, nx::Integer, stride::Integer = nbatch)
    nbatch >= 0 || throw(ArgumentError("nbatch must be non-negative"))
    nx >= 0 || throw(ArgumentError("nx must be non-negative"))
    stride >= nbatch || throw(ArgumentError("stride must be at least nbatch"))
    all(x -> x <= typemax(UInt32), (nbatch, nx, stride)) ||
        throw(ArgumentError("Vulkan prototype dimensions must fit in UInt32"))
    ThomasPushConstants(UInt32(nbatch), UInt32(nx), UInt32(stride))
end

"""Host-independent launch geometry for one work item per independent system."""
struct ThomasDispatch
    parameters::ThomasPushConstants
    workgroupsize::UInt32
    ngroups::UInt32
end

function ThomasDispatch(nbatch::Integer, nx::Integer;
                        stride::Integer = nbatch, workgroupsize::Integer = 256)
    workgroupsize > 0 || throw(ArgumentError("workgroupsize must be positive"))
    workgroupsize <= typemax(UInt32) ||
        throw(ArgumentError("workgroupsize must fit in UInt32"))
    parameters = ThomasPushConstants(nbatch, nx, stride)
    ngroups = cld(nbatch, workgroupsize)
    ngroups <= typemax(UInt32) || throw(ArgumentError("group count must fit in UInt32"))
    ThomasDispatch(parameters, UInt32(workgroupsize), UInt32(ngroups))
end

