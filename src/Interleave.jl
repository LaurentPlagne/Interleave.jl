"""
    Interleave

Apply an intrinsically sequential algorithm—a recurrence—to a batch of independent
problems. Data Layout Interleaving exposes SIMD parallelism across that batch, with optional
and explicit task parallelism as a second level.

Develop the algorithm on ordinary multidimensional `Base.Array` objects. Write the kernel
once in scalar notation; changing the element type through the `P` parameter of
[`Interleave.Array`](@ref) selects the packed specialization.

The packed type expresses explicit SIMD lanes for supported operations. It does not
guarantee optimal performance: the best `P` depends on the kernel and target machine and
must be measured.

Do not use `@fastmath` in a kernel that must match its scalar reference exactly. FMA
contraction can break bit-exact agreement.
"""
module Interleave

using SIMD: Vec, VecTypes, vifelse
using OhMyThreads: tforeach, index_chunks, SerialScheduler, StaticScheduler,
                   DynamicScheduler, GreedyScheduler

# `Array` n'est PAS exporté : il porte le même nom que `Base.Array` et s'écrit
# `Interleave.Array`, exactement comme `Interleave::Array` en C++.
export packet, instance, packs, scratchlike, tune,
       apply!, parallel_apply!,
       gpu_apply!, gpu_backend, gpu_synchronize, gpu_scratchlike,
       packsize, npacks, instance_size, npadding, lanetype, packtype
# Réexportés par commodité : ce sont les schedulers attendus par `parallel_apply!`.
export SerialScheduler, StaticScheduler, DynamicScheduler, GreedyScheduler
#  est la seule façon d écrire une condition qui vaut pour un paquet ET pour un
# scalaire : c est la primitive qui préserve la promesse « un seul noyau ».
export Vec, vifelse

include("array.jl")
include("apply.jl")
include("repack.jl")
include("tune.jl")
include("gpu.jl")

end # module
