# Dimensions and the packed axis

The most important layout decision is not packet size. It is choosing what one lane means.

## The first dimension is the population

For every Interleave batch:

```text
size(A) == (number of jobs, dimensions of one job...)
```

Examples:

| application | logical shape | one lane represents |
|---|---|---|
| filter bank | `(channels, samples)` | one channel |
| tridiagonal ensemble | `(systems, unknowns)` | one linear system |
| image-channel operation | `(channels, height, width)` | one channel |
| parameter study | `(parameters, state_variables)` | one parameter set |
| x-sweep in ADI | `(y_lines, x, z)` | one y-position |

Interleave packs consecutive values of the first logical dimension. The remaining dimensions
become the dense instance presented to the kernel.

## Choose independence before locality

The packed axis must identify independent jobs. If lane 3 reads or writes lane 4, the model
is wrong. Once independence is established, prefer an axis that:

- supplies enough jobs to fill packets;
- gives jobs similar control flow and runtime;
- lets the packed layout live across several kernel calls;
- minimizes transposition or packing cost.

The largest axis is not automatically best. A smaller but more uniform population may waste
less masked or divergent work.

## Column-major Julia and the scalar oracle

The batch-first logical convention makes a standard Julia instance such as
`view(A, job, :)` strided. This is intentional: the public shape remains identical when the
type changes, while `Interleave.Array` rearranges physical storage so a packed instance is
contiguous.

This distinction matters when benchmarking. Compare full application paths and state the
layout of the scalar reference. A batch-last scalar array may be a faster hand-optimized
reference, but it no longer has the same constructor and indexing contract.

## Multidimensional instances

An instance can be a matrix or a volume. `packet(A, k)` removes only the packet dimension:

```@example dimensions
using Interleave

A = Interleave.Array{Float32,4,8}(undef, 33, 10, 12, 14)
(size(A), instance_size(A), size(instance(A, 1)), size(parent(A)))
```

The kernel should traverse those dimensions in Julia's natural column-major order whenever
the recurrence permits it.

## Partial packets

For `nbatch=33` and `P=8`, Interleave allocates five packets and seven padding lanes. The logical
size remains 33. Padding affects memory and executed instructions but never scalar indexing
or reductions.

If partial packets dominate—say, three jobs with `P=16`—choose a smaller `P` or use the
standard array path.

