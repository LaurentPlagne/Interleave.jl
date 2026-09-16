"""Run every validation kernel through the generic KernelAbstractions driver.

The functions below are the same scalar Julia kernels used by `apply!` and the CPU test
suite.  Only the storage changes from `Array` to a backend device array; no vendor shader
is written for an individual algorithm.  Set `INTERLEAVE_KA_BACKEND` to `metal`, `cuda`,
or `amdgpu` to select the backend.  The sizes are intentionally resident and modest
enough for GitHub runners.
"""

const KA_BACKEND = lowercase(get(ENV, "INTERLEAVE_KA_BACKEND", "metal"))

if KA_BACKEND == "metal"
    using Metal
    const DeviceArray = Metal.MtlArray
    const backend_functional = Metal.functional
elseif KA_BACKEND == "cuda"
    using CUDA
    const DeviceArray = CUDA.CuArray
    const backend_functional = CUDA.functional
elseif KA_BACKEND == "amdgpu"
    using AMDGPU
    const DeviceArray = AMDGPU.ROCArray
    const backend_functional = AMDGPU.functional
else
    error("unsupported INTERLEAVE_KA_BACKEND=$KA_BACKEND (expected metal, cuda, or amdgpu)")
end

using Interleave
using KernelAbstractions
using BenchmarkTools

include(joinpath(@__DIR__, "..", "..", "test", "kernels.jl"))

const BACKEND_AVAILABLE = backend_functional()

const T = Float32

function _host_reference(f, host, scratch)
    result = copy(first(host))
    args = (result, Base.tail(host)...)
    if scratch === nothing
        apply!(f, args...)
    else
        # CPU `apply!` receives one instance-sized workspace; the GPU driver
        # receives a batch-major workspace and selects one row per work item.
        # Do not accidentally validate the CPU path with a linear slice of the
        # batched GPU scratch buffer.
        instance_scratch = view(scratch, 1,
                                ntuple(_ -> Colon(), Val(ndims(scratch) - 1))...)
        apply!(f, args...; scratch = copy(instance_scratch))
    end
    result
end

function _run_gpu(f, device, scratch)
    if scratch === nothing
        gpu_apply!(f, device...; wait = true)
    else
        gpu_apply!(f, device...; scratch = scratch, wait = true)
    end
    nothing
end

function _case(name, f, host; scratch = nothing, rtol = 5f-5)
    reference = _host_reference(f, host, scratch)
    device = map(DeviceArray, host)
    dscratch = scratch === nothing ? nothing : DeviceArray(scratch)
    _run_gpu(f, device, dscratch)
    got = Array(first(device))
    err = maximum(abs, got .- reference)
    scale = max(maximum(abs, reference), 1f0)
    err <= rtol * scale || error("$name: max error $err exceeds $(rtol * scale)")

    # Reset all device buffers before every sample, outside the timed region.  Several
    # kernels are in-place recurrences; without this setup each sample would solve the
    # output of the preceding sample rather than the same problem.
    pristine = map(DeviceArray, host)
    reset = () -> begin
        for i in eachindex(device)
            copyto!(device[i], pristine[i])
        end
        dscratch === nothing || fill!(dscratch, zero(eltype(dscratch)))
        nothing
    end
    trial = BenchmarkTools.@benchmarkable _run_gpu($f, $device, $dscratch) setup = ($reset())
    elapsed = BenchmarkTools.minimum(BenchmarkTools.run(trial; samples = 5)).time / 1e9
    println(rpad(name, 22), " ", lpad(round(elapsed * 1e3; digits = 3), 9),
            " ms | max error ", err)
    nothing
end

function main()
    if !BACKEND_AVAILABLE
        println("$(KA_BACKEND) KernelAbstractions suite skipped (no functional GPU on this runner)")
        return nothing
    end
    nb, nx = 4_096, 64
    X = fill(T(1), nb, nx); D = fill(T(2), nb, nx)
    U = fill(T(-1), nb, nx); L = fill(T(-1), nb, nx)
    B = [sinpi(T(b) / 8) + T(i) / nx for b in 1:nb, i in 1:nx]
    _case("Thomas", thomas!, (zeros(T, nb, nx), D, U, L, B);
          scratch = zeros(T, nb, nx))

    nb, ns = 1_024, 256
    _case("Biquad", gpu_biquad!, (zeros(T, nb, ns), fill(T(1), nb, ns)))

    nb, H, W = 256, 64, 64
    _case("Depthwise 3x3", gpu_depthwise3x3!,
          (zeros(T, nb, H, W), fill(T(1), nb, H, W)))
    _case("Sobel + motion", gpu_sobel_motion!,
          (zeros(T, nb, H, W), fill(T(1), nb, H, W), fill(T(0.5), nb, H, W)))

    nopt, ngrid = 1_024, 32
    V = [max(T(i) - T(b) / 100, 0) for b in 1:nopt, i in 1:ngrid]
    D = fill(T(2.05), nopt, ngrid); U = fill(T(-0.5), nopt, ngrid)
    L = fill(T(-0.5), nopt, ngrid); R = zeros(T, nopt, ngrid)
    _case("Black-Scholes CN", gpu_blackscholes_cn!, (V, D, U, L, R);
          scratch = zeros(T, nopt, ngrid), rtol = 2f-4)

    nb, n1, n2, n3 = 128, 24, 24, 16
    I3 = [T(b) + T(i) / 10 + T(j) / 100 + T(k) / 1000
          for b in 1:nb, i in 1:n1, j in 1:n2, k in 1:n3]
    _case("Laplacian 3D", laplacien3d!, (zeros(T, nb, n1, n2, n3), I3))

    nb, nx, m = 256, 32, 3
    D = fill(T(2), nb, nx, m); U = fill(T(-1), nb, nx, m)
    L = fill(T(-1), nb, nx, m)
    B = [sinpi(T(b) / 8) + T(i + c) / nx for b in 1:nb, i in 1:nx, c in 1:m]
    _case("Thomas lines", thomas_lines!, (zeros(T, nb, nx, m), D, U, L, B);
          scratch = zeros(T, nb, nx))

    _case("Tridiagonal product", tridiag_mul!,
          (zeros(T, nb, nx), D[:, :, 1], U[:, :, 1], L[:, :, 1], fill(T(0.25), nb, nx)))

    nb, n = 1_024, 256
    A = [sinpi(T(b) / 17) + T(i) / n for b in 1:nb, i in 1:n]
    B = [cospi(T(b) / 23) - T(i) / (2n) for b in 1:nb, i in 1:n]
    _case("Squared norm", batch_squarednorm!, (zeros(T, nb, 1), A))
    _case("Dot product", batch_dot!, (zeros(T, nb, 1), A, B))
    println("$(KA_BACKEND) KernelAbstractions suite passed")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
