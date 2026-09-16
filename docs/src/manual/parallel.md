# Explicit task parallelism

SIMD and task parallelism are separate decisions.

```julia
apply!(kernel!, output, input)  # sequential packet traversal

parallel_apply!(kernel!, output, input;
                scheduler=StaticScheduler())
```

[`apply!`](@ref) cannot launch tasks: it has no scheduler or chunk-count keyword. This makes
it safe inside a caller that already owns parallelism. [`parallel_apply!`](@ref) makes task
creation visible at the call site.

## Choose a scheduler

The exported OhMyThreads schedulers cover common policies:

- `StaticScheduler()` for uniform packet cost;
- `DynamicScheduler()` or `GreedyScheduler()` for uneven work;
- `SerialScheduler()` to exercise the parallel driver without concurrent tasks.

The default number of chunks is `4 * Threads.nthreads()`. Override `nchunks` only after
measuring; too many chunks increase launch overhead, while too few reduce load balance.

## Scratch storage is per chunk

When a prototype array is supplied, the driver calls `similar` once per chunk:

```julia
parallel_apply!(kernel!, output, input;
                scratch=scratchlike(output),
                scheduler=StaticScheduler())
```

Never return the same mutable scratch object to every task. That creates a race even when
each packet writes to independent output.

## Expect bandwidth ceilings

Threads multiply throughput only while cores have independent resources available. A
compute-heavy recurrence that reuses cached state may scale well. A solver streaming five
large arrays can saturate memory bandwidth after only a few cores.

Measure the best sequential `P` first, then add tasks. Reporting SIMD speedup multiplied by
thread count is not a substitute for an end-to-end measurement.

## Test determinism

Run the parallel result several times and compare each run exactly with the sequential
result. Reproducibility alone is insufficient—a deterministic indexing error can repeat.
Agreement with the scalar oracle, agreement with sequential DLI, and repeated parallel
agreement test different failure modes.

