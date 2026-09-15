# Writing kernels

A kernel operates on one scalar problem or one packet of problems. It never loops over the
batch.

## The basic contract

```julia
function kernel!(output, input, parameters...)
    # iterate only over dimensions of one instance
    output
end

apply!(kernel!, output_batch, input_batch)
```

All batch arrays passed to one call must have the same logical batch size and packet size.
Their instance shapes may differ when the algorithm intentionally uses different operands.

## Let values carry their type

Avoid hard-coded vector types in kernel code. Derive state from array elements:

```julia
state = zero(eltype(output))
coefficient = lanetype(eltype(output))(0.125)
```

`lanetype` is useful for numeric literals because a scalar `Float32` can multiply either a
`Float32` or a `Vec{P,Float32}`. It avoids accidentally introducing `Float64` literals.

## Preserve operation order

Bit-exact scalar-to-packed comparison relies on executing the same operations in the same
order. Therefore:

- do not use `@fastmath`;
- do not reorder sums for convenience;
- do not replace division with a differently ordered reciprocal expression;
- use `muladd` explicitly in both versions if fused multiply-add is intended.

This requirement is stronger than normal approximate numerical testing, but it catches lane
mixups, layout errors, and padding leaks with a single oracle.

## Keep lane control flow uniform

Good kernels use lane-varying values with a common loop structure. Poor fits include:

- lane-dependent loop bounds;
- early exit for only some jobs;
- frequent branches selected by packed values;
- cross-lane reductions inside the recurrence.

Some branches can be expressed with SIMD selection, but doing so is algorithm-specific and
can waste most work when lanes diverge.

A direct branch on packed data normally fails early rather than silently choosing one lane:
`x[i] > 0` produces `Vec{P,Bool}`, which is not the scalar `Bool` required by `if`. When both
arms are safe to evaluate, `SIMD.vifelse(mask, a, b)` can select lane by lane, but both arms
still contribute work. A branch controlled by a scalar parameter or an index is uniform and
is fine.

The library cannot reliably “ban divergence” by inspecting Julia syntax. Divergence can be
hidden behind a called function, and masked selection may be exactly the intended algorithm.
The practical policy is: reject unsupported packed control flow through compilation tests,
document intentional `vifelse`, and benchmark it. Lane-dependent loop counts or side effects
remain outside the safe single-kernel model.

## Allocate scratch once per driver call

Never allocate work arrays inside a kernel invoked once per packet. Pass a prototype through
the driver:

```julia
scratch = similar(instance(output, 1))
apply!(kernel!, output, input; scratch)
```

The sequential driver creates one scratch array. The parallel driver creates one per chunk,
preventing races without allocating once per packet. An explicit zero-argument callable is
also accepted when construction needs more control.

## Demand inference and zero hot-path allocations

Useful regression checks are:

```julia
@inferred instance(A, 1)
@inferred apply!(kernel!, output, input)
@allocated apply!(kernel!, output, input) == 0
```

Warm the call before measuring allocations. Keep the measurement inside a function so global
variables do not introduce boxing or dynamic dispatch.

## Supported element operations define the scope

The packed specialization only works when every operation used by the kernel exists for
`SIMD.Vec`. Basic arithmetic and many elementary patterns work naturally; arbitrary Julia
objects, dynamically typed values, and automatic-differentiation dual numbers inside a
`Vec` do not currently form a general solution.
