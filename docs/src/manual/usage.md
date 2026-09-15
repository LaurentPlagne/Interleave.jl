# Working with a batch

## Construct it

`Interleave.Array` follows the parameter order of `Base.Array` and adds packet size `P`:

```@example usage
using Interleave

A = Interleave.Array{Float32,2,8}(undef, 61, 32)
(size(A), eltype(A), packsize(A), npacks(A), npadding(A))
```

The first dimension is always the batch. Every later dimension belongs to one logical
problem. Packet size must be a power of two.

An existing scalar array—including the result of a comprehension—can be packed directly:

```@example usage
C = Interleave.Array{Float32,2,8}([
    Float32(job) + Float32(i) / 10 for job in 1:13, i in 1:6
])
(size(C), packsize(C), C[3, 2])
```

This convenient constructor materializes the scalar source and then copies it into packed
storage. Prefer `undef` plus broadcast or an initialization loop when that temporary would
matter.

When `P` is a runtime choice, use the keyword constructor:

```@example usage
P = 4
B = Interleave.Array{Float32}(undef, 17, 6, 5; pack=Val(P))
(size(B), instance_size(B))
```

## Fill and inspect it as scalars

A Interleave batch is an `AbstractArray{T}`. Normal scalar indexing, broadcasting, reductions,
views, copying, and collection operate on the logical shape:

```@example usage
fill!(A, 0)
for job in axes(A, 1), i in axes(A, 2)
    A[job, i] = job + i / 100
end

(A[3, 7], sum(@view A[1:2, :]), Array(A) isa Matrix{Float32})
```

Padding is not part of this logical view and cannot leak into `sum(A)` or `Array(A)`.

## Give packed instances to a kernel

`instance(A, k)` exposes the `k`th packet to the hot kernel:

```@example usage
v = instance(A, 1)
(size(v), strides(v), eltype(v))
```

Most callers should use [`apply!`](@ref) rather than loop over packets themselves:

```julia
apply!(kernel!, output, input)
```

For composition with Julia's standard iteration vocabulary, [`packs`](@ref) returns a lazy
vector of instance views:

```julia
foreach(kernel!, packs(output), packs(input))
```

Both forms are sequential.

## Start with a standard array

A `Base.Array` is a valid batch with packet size one:

```@example usage
S = zeros(Float32, 12, 20)
(packsize(S), npacks(S), instance_size(S), eltype(instance(S, 1)))
```

This is the recommended development path: make the kernel correct with ordinary arrays,
then change the type. Because Julia is column-major, `instance(S, 1)` is strided when the
batch is first, whereas a Interleave packed instance is contiguous. The standard path is an
oracle and a `P=1` option, not necessarily the fastest scalar storage for every application.

## Understand logical and physical shapes

For `A = Interleave.Array{Float32,3,8}(undef, 61, 6, 5)`:

| expression | result | meaning |
|---|---|---|
| `size(A)` | `(61, 6, 5)` | scalar user view |
| `eltype(A)` | `Float32` | scalar user element |
| `size(parent(A))` | `(6, 5, 8)` | packed physical storage |
| `eltype(parent(A))` | `Vec{8,Float32}` | kernel element |
| `size(instance(A, 1))` | `(6, 5)` | one packed problem view |

`parent(A)` is intentionally public for low-level inspection and integration. Kernels
normally receive its views from the driver.

## Empty and partial batches

An empty batch is valid. `apply!`, `parallel_apply!`, `packs`, and `foreach` perform no
kernel calls. A non-multiple batch allocates one partial final packet; its unused lanes are
initialized to zero and hidden from the logical array.

The padding lanes still execute the kernel. Avoid kernels for which even a zero-initialized
dummy lane triggers an invalid operation. If validity depends on per-job data, initialize
all kernel inputs so the padded computation stays in-domain.

## Wrap existing packed storage

If data already has the native physical representation, it can be wrapped without copying:

```@example usage
storage = fill(Vec{4,Float32}(0), 32, 3)
C = Interleave.Array(storage, 10)
(size(C), parent(C) === storage, npadding(C))
```

The constructor clears unused lanes in the last packet to preserve the padding invariant.
