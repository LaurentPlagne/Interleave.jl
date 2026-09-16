# Per-instance reductions

The VulkanBench example also measures squared norms and dot products. They are useful here
as a control case: the reduction is sequential *inside one instance*, while independent
instances remain available for SIMD or GPU work.

```julia
function batch_squarednorm!(out, A)
    acc = zero(eltype(A))
    @inbounds for i in eachindex(A)
        x = A[i]
        acc += x * x
    end
    out[1] = acc
    out
end
```

The same function is passed to all three drivers:

```julia
apply!(batch_squarednorm!, out, A)
parallel_apply!(batch_squarednorm!, out, A)
gpu_apply!(batch_squarednorm!, d_out, d_A; wait = true)
```

Only the container and execution backend change. There is no reduction-specific GPU shader
in Interleave.jl. Each GPU work item owns one vector and writes one result, so the batch axis
is still coalesced. The reduction is intentionally included as a control workload: for a
large contiguous scalar vector, ordinary compiler SIMD may already be excellent, and a
large packet size is not automatically beneficial.

The benchmark is part of the common suite:

```sh
julia --project=bench -t auto bench/reductions.jl
```

The Metal workflow runs the same two kernels through KernelAbstractions. Its result is a
source-reuse and correctness check across backends, not a claim that one work item per
reduction is the best possible vendor-specific reduction implementation.
