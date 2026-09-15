# API reference

```@docs
Interleave
```

## Batch representation

```@docs
Interleave.Array
packtype
packsize
npacks
npadding
instance_size
lanetype
Base.parent(::Interleave.Array)
```

## Traversal

```@docs
instance
packs
apply!
parallel_apply!
```

## GPU execution

```@docs
gpu_apply!
gpu_backend
gpu_synchronize
```

## Schedulers

Interleave re-exports these scheduler types from OhMyThreads for use with
[`parallel_apply!`](@ref):

- `SerialScheduler`
- `StaticScheduler`
- `DynamicScheduler`
- `GreedyScheduler`

See the [OhMyThreads scheduler documentation](https://juliafolds2.github.io/OhMyThreads.jl/stable/refs/api/#Schedulers)
for their complete option reference.
