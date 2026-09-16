# Tutorial 3 — Repacking a 3-D ADI solver

Alternating-direction implicit schemes solve many independent tridiagonal lines along one
grid direction, then another. The recurrence direction changes, but the line solver does
not have to.

![Changing recurrence and packed axes during ADI sweeps](../assets/adi-axis.svg)

## 1. Separate physical axes from kernel axes

Let the physical grid be `(nx, ny, nz)`. During an x-sweep:

- each recurrence runs along `x`;
- different `y` or `z` lines are independent;
- one independent grid axis can become the Interleave batch dimension.

For example, packing across `y` gives a logical batch of shape `(ny, nx, nz)`:

```@example adi
using Interleave

nx, ny, nz = 32, 20, 12
X = Interleave.Array{Float32,3,8}(undef, ny, nx, nz)
(size(X), instance_size(X), size(parent(X)))
```

The logical indices are `X[y, x, z]`. A kernel instance has shape `(nx, nz)` and elements
of type `Vec{8,Float32}`. It can solve every x-line in that instance with ordinary `(x,z)`
indices.

## 2. Write a line solver, not an axis-specific solver

```@example adi
function thomas_lines!(x, d, u, l, b, scratch)
    n, nlines = size(x)
    @inbounds for line in 1:nlines
        pivot = d[1, line]
        invpivot = inv(pivot)
        x[1, line] = b[1, line] * invpivot
        for i in 2:n
            scratch[i] = u[i - 1, line] * invpivot
            pivot = d[i, line] - l[i, line] * scratch[i]
            x[i, line] = b[i, line] - l[i, line] * x[i - 1, line]
            invpivot = inv(pivot)
            x[i, line] *= invpivot
        end
        for i in n-1:-1:1
            x[i, line] -= scratch[i + 1] * x[i + 1, line]
        end
    end
    x
end
```

The solver only knows that its first local dimension is the recurrence direction. Layout
decides whether that dimension represents physical `x`, `y`, or `z`.

## 3. Prepare one sweep

```@example adi
X, D, U, L, B = (Interleave.Array{Float32,3,8}(undef, ny, nx, nz) for _ in 1:5)
fill!(X, 0); fill!(D, 2); fill!(U, -1); fill!(L, -1); fill!(B, 1)

apply!(thomas_lines!, X, D, U, L, B;
       scratch=Vector{packtype(X)}(undef, nx))
size(X)
```

The scratch buffer has one entry per recurrence position and uses the packed element type.
For parallel execution, pass the same object as a prototype; the driver creates one copy per
chunk.

## 4. Change direction by repacking

For a y-sweep, the batch axis must become `x`. The logical ordering `(nx, ny, nz)` does it:
`x` supplies lanes, each kernel instance has shape `(ny, nz)`, and the recurrence runs along
`ny`.

Moving the data there is [`permutedims!`](@ref Base.permutedims!), the ordinary Julia verb:

```julia
Y = Interleave.Array{Float32,3,8}(undef, nx, ny, nz)
permutedims!(Y, X, (2, 1, 3))        # Y[i, y, k] == X[y, i, k]
```

Earlier revisions of this tutorial described this step without providing it, which left the
generic fallback — correct, but reaching every element through scalar indexing and paying a
`divrem` per element to find its lane. Interleave now specialises it.

Two regimes matter, and the cheaper one is easy to miss:

- **the batch axis stays put** (`perm[1] == 1`, such as `(1, 3, 2)`): no lane crosses a
  packet, so this is a plain `permutedims!` on the packed storage;
- **the batch axis moves** (`(2, 1, 3)` or `(3, 2, 1)`): lanes must cross packets, which
  becomes a `P × P` tile transpose.

If both remaining grid axes are independent, prefer the permutation that keeps the batch axis
in place.

## 5. What repacking actually costs

Measured on a 128×128×64 grid, `P = 8`, one core of an M1 Max:

| | time | in Thomas sweeps |
|---|---:|---:|
| one line-solve sweep | 1.71 ms | 1.00 |
| repack, generic fallback | 2.13 ms | 1.25 |
| **repack, specialised** | **0.47 ms** | **0.27** |
| `permutedims!` on an equivalent `Base.Array` | 0.40 ms | 0.23 |

The last row is the floor: the same bytes moved, with no packets involved. The specialised
path lands 18% above it, so there is little left to win on this operation.

For a full ADI stage — three sweeps and two repacks:

| | stage | share spent repacking |
|---|---:|---:|
| generic fallback | 9.39 ms | **45%** |
| specialised | 6.06 ms | 15% |

Repacking was quietly consuming nearly half the stage and is now a sixth of it, a **35%
improvement on the whole stage** for a change no kernel sees.

!!! tip "Is it worth fusing the transpose into the solve?"
    A natural next idea is to skip the separate pass entirely: read from the previous layout
    and write straight into the next one. The table above bounds what that could buy. Even a
    *perfect* fusion, with the repack costing literally nothing, would take 6.06 ms to 5.13 ms
    — **15% more**.

    And that bound is optimistic. Fusing means the sweep reads or writes across the packed
    axis, so the contiguous packet loads that make DLI fast become strided, and the sweep
    itself slows down. It also costs the property the whole package is built on: the kernel
    would have to know about two layouts instead of none.

    Measure before assuming otherwise, but on this evidence the separate repack is the right
    trade: it captured 35%, and it leaves at most 15% on the table for a large loss of
    simplicity.

Repacking is paid once; the recurrence may run for many time steps or nonlinear iterations.
The approach is compelling when the packed layout is reused enough times to amortize the data
movement. It is weak when every solve is tiny and a full transpose is required before and
after each call. Measure the complete stage:

```text
reorder → repeated line solves → reorder back
```

Do not report the line solver alone if the application cannot keep data in the packed layout.
