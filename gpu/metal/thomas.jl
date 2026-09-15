using Interleave
using Metal
using BenchmarkTools

# Use the same scalar kernel as the CPU tests and the all-kernel Metal suite.  The
# KernelAbstractions driver, rather than a hand-written Metal shader, supplies the
# one-work-item-per-batch mapping.
include(joinpath(@__DIR__, "..", "..", "test", "kernels.jl"))

function main(; nbatch = 65_536, nx = 64, workgroupsize = 256)
    Metal.functional() || error("Metal.jl did not find a supported Apple GPU")

    host(v) = fill(Float32(v), nbatch, nx)
    X, D, U, L, B = host(0), host(2), host(-1), host(-1), host(1)

    # Device buffers keep batch as their fastest-varying dimension.
    dX, dD, dU, dL, dB = MtlArray.((X, D, U, L, B))
    dS = similar(dX)

    gpu_apply!(thomas!, dX, dD, dU, dL, dB;
               scratch = dS, workgroupsize, wait = true)

    result = Array(dX)
    reference = copy(X)
    apply!(thomas!, reference, D, U, L, B; scratch = zeros(Float32, nx))
    @assert result == reference
    elapsed = @belapsed gpu_apply!($(thomas!), $dX, $dD, $dU, $dL, $dB;
                                   scratch = $dS, workgroupsize = $workgroupsize,
                                   wait = true) samples = 10 evals = 1
    println("Metal Thomas: ", round(elapsed * 1e3; digits = 3),
            " ms | ", round(2 * nbatch * nx / elapsed / 1e9; digits = 2),
            " GF32/s | workgroup = ", workgroupsize)
    result
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
