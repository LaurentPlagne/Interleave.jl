"""Backend-neutral KernelAbstractions benchmark entry point.

The implementation is shared with the historical Metal path.  Select the backend with
`INTERLEAVE_KA_BACKEND=metal|cuda|amdgpu`.
"""

include(joinpath(@__DIR__, "..", "metal", "all.jl"))
