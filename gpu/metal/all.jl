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

# Données non uniformes, déterministes, variant selon **tous** les axes — le lot compris.
#
# Les entrées constantes qu'utilisait cette suite rendaient plusieurs cas vides de contenu :
# un Sobel d'image constante vaut zéro partout, et une image constante est invariante par
# transposition, donc une inversion d'indices dans la vue device passait inaperçue. La
# comparaison au calcul CPU était alors satisfaite par deux tableaux nuls.
function _varied(dims::Vararg{Int,N}; seed = 0) where {N}
    w = ntuple(i -> T(2i - 1), Val(N))
    [T(sinpi((sum(w .* Tuple(I)) + seed) / 23)) for I in CartesianIndices(dims)]
end

"""Système tridiagonal batch-major non uniforme, à diagonale strictement dominante."""
function _varied_tridiag(dims::Vararg{Int,N}; seed = 0) where {N}
    D = T(4) .+ _varied(dims...; seed = seed + 1)          # |D| ≥ 3
    U = T(-1) .+ T(0.25) .* _varied(dims...; seed = seed + 2)   # |U| + |L| ≤ 2.5
    L = T(-1) .+ T(0.25) .* _varied(dims...; seed = seed + 3)
    B = _varied(dims...; seed = seed + 4)
    D, U, L, B
end

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
    # Une référence identiquement nulle satisferait la comparaison ci-dessus quel que soit
    # le résultat GPU. Le cas s'est produit tant que les entrées étaient constantes.
    any(!iszero, reference) ||
        error("$name: the CPU reference is identically zero, so the check proves nothing")

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
    # Tous les lots sont volontairement **non multiples** du `workgroupsize` (256 par
    # défaut) : c'est le cas qui exerce la garde `batch <= nbatch` du noyau de lancement,
    # et celui qu'aucune taille de cette suite ne touchait auparavant.
    nb, nx = 4_093, 64
    D, U, L, B = _varied_tridiag(nb, nx)
    _case("Thomas", thomas!, (zeros(T, nb, nx), D, U, L, B);
          scratch = zeros(T, nb, nx))

    nb, ns = 1_021, 256
    _case("Biquad", gpu_biquad!, (zeros(T, nb, ns), _varied(nb, ns; seed = 10)))

    # H ≠ W : une transposition d'indices dans la vue device ne peut plus passer.
    nb, H, W = 251, 64, 48
    _case("Depthwise 3x3", gpu_depthwise3x3!,
          (zeros(T, nb, H, W), _varied(nb, H, W; seed = 11)))
    _case("Sobel + motion", gpu_sobel_motion!,
          (zeros(T, nb, H, W), _varied(nb, H, W; seed = 12),
           _varied(nb, H, W; seed = 13)))

    nopt, ngrid = 1_019, 32
    V = [max(T(i) - T(b) / 100, 0) for b in 1:nopt, i in 1:ngrid]
    D = T(2.05) .+ T(0.2) .* _varied(nopt, ngrid; seed = 14)      # |D| ≥ 1.85
    U = T(-0.5) .+ T(0.1) .* _varied(nopt, ngrid; seed = 15)      # |U| + |L| ≤ 1.2
    L = T(-0.5) .+ T(0.1) .* _varied(nopt, ngrid; seed = 16)
    R = zeros(T, nopt, ngrid)
    _case("Black-Scholes CN", gpu_blackscholes_cn!, (V, D, U, L, R);
          scratch = zeros(T, nopt, ngrid), rtol = 2f-4)

    # Trois extents distincts : un échange d'axes est détectable par la forme elle-même.
    nb, n1, n2, n3 = 127, 24, 20, 16
    _case("Laplacian 3D", laplacien3d!,
          (zeros(T, nb, n1, n2, n3), _varied(nb, n1, n2, n3; seed = 17)))

    nb, nx, m = 253, 32, 3
    D, U, L, B = _varied_tridiag(nb, nx, m; seed = 20)
    _case("Thomas lines", thomas_lines!, (zeros(T, nb, nx, m), D, U, L, B);
          scratch = zeros(T, nb, nx))

    _case("Tridiagonal product", tridiag_mul!,
          (zeros(T, nb, nx), D[:, :, 1], U[:, :, 1], L[:, :, 1],
           _varied(nb, nx; seed = 24)))

    nb, n = 1_021, 256
    A = _varied(nb, n; seed = 25)
    B = _varied(nb, n; seed = 26)
    _case("Squared norm", batch_squarednorm!, (zeros(T, nb, 1), A))
    _case("Dot product", batch_dot!, (zeros(T, nb, 1), A, B))
    println("$(KA_BACKEND) KernelAbstractions suite passed")
end

# Lancé directement, on exécute. Inclus par `gpu/ka/all.jl`, c'est lui qui appelle `main()` —
# d'où la garde, qui évite de le faire deux fois.
abspath(PROGRAM_FILE) == (@__FILE__) && main()
