# What you would write instead

[Positioning and alternatives](alternatives.md) surveys the tools. This page answers the
narrower and more useful question, kernel by kernel: **if Interleave did not exist, what would
you actually write, and would you be worse off?**

Two of the six answers are "you would be better off". Those are the important ones.

## Where the numbers come from

Everything below was measured by `bench/runall.jl` on an Apple M-series machine (NEON,
128-bit vectors, 8 threads, `Float32`). The *reference* is the naive Julia loop over a
`Base.Array` batch — the code you would write first, with no annotation.

| kernel | recurrence? | reference throughput | best DLI | with threads |
|---|:---:|---:|---:|---:|
| Thomas tridiagonal | yes | 1.1 GFlop/s | `P=16`: **13.4×** | 31.8× |
| IIR biquad | yes | 3.0 GFlop/s | `P=16`: **15.0×** | 32.9× |
| Black–Scholes CN | yes (double) | 1.9 GFlop/s | `P=16`: **11.8×** | 84.6× |
| Depthwise 3×3 | **no** | 40.6 GFlop/s | `P=16`: **0.43×** | 1.73× |
| Sobel + motion | **no** | 43.4 GFlop/s | `P=8`: **0.88×** | 2.77× |

!!! tip "The decision rule is the reference throughput, not the application domain"
    A reference at 1–3 GFlop/s means the compiler has failed — there is a recurrence, and
    DLI returns 12–15×. A reference at 40+ GFlop/s means LLVM already vectorized the inner
    loop; DLI can then only *take over* that gain, never add to it, and it takes it over
    badly.

    The mechanism is worth stating plainly: **`Vec{P,T}` as an element type blocks the
    auto-vectorization LLVM would have performed by itself.** On the two stencil kernels,
    `P=1` is 4–10× *faster* than `P=16`, because a one-lane vector element prevents
    vectorization along the contiguous axis. On the recurrence kernels, `P=1` tracks the
    reference to within 2% — there was nothing to block.

## 1. Thomas tridiagonal solve — Interleave wins

**Without Interleave** you would loop over systems and call the scalar sweep, or reach for
`LinearAlgebra.Tridiagonal`. Neither helps:

- `Tridiagonal` plus `ldiv!` dispatches to LAPACK's `gtsv` **one system at a time**. For
  65 536 systems of length 64, per-call overhead dominates and there is no batching.
- `@simd` on the forward sweep is simply illegal: `X[i]` depends on `X[i-1]`. The
  [performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/) warn that the
  macro promises reorderable iterations and produces wrong results otherwise.
- `LoopVectorization.@turbo` assumes iteration independence, so it is illegal on the same
  axis for the same reason.

The one alternative that *does* work is to transpose your data into a global SoA layout and
rewrite the kernel as "time outside, instances inside". That is DLI done by hand — and it
costs you the readable one-problem kernel, plus cache locality once the population is large,
since `i-1` is then `nbatch` scalars away.

**Verdict.** 13.4× at `P=16`, and [`tune`](@ref) finds 18.3× at `P=32` on this machine — the
packet size is genuinely worth measuring rather than assuming.

*Strength:* the kernel source is unchanged and stays bit-exact against its scalar self.
*Weakness:* you own a tuning parameter you did not have before.

## 2. IIR biquad filter bank — complementary, not competing

**Without Interleave** you would use [DSP.jl](https://docs.juliadsp.org/stable/filters/) and
`filt`. That is a real, well-optimized answer — for **one** signal. A bank of hundreds of
independent filters is a loop around it, and each call carries the same strict temporal
recurrence that no compiler vectorizes.

DSP.jl also gives you everything Interleave does not: filter *design* (Butterworth,
Chebyshev), form conversion, stability analysis. Interleave has none of that and should not
try.

**Verdict.** 15.0×, the largest single-threaded gain of the suite. The honest recommendation
is to use both: **design the filter with DSP.jl, run the bank with Interleave.**

## 3. Black–Scholes Crank–Nicolson — the best case, at a price

**Without Interleave** you would either hand-roll the scheme (what the benchmark's reference
does) or use [MethodOfLines.jl](https://docs.sciml.ai/MethodOfLines/dev/) and the SciML
stack.

SciML buys adaptivity, event handling, sensitivity analysis, and a solver ecosystem.
Interleave buys none of that. But once you have *committed* to a fixed scheme — Crank–Nicolson,
fixed grid, fixed number of steps — that machinery is overhead on a kernel that is now just
two nested recurrences.

**Verdict.** 11.8× sequential and **84.6× with threads** — the best result in the suite,
because the double recurrence (time × sweep) is doubly hostile to auto-vectorization, and the
working set stays small enough that threads still scale.

*Weakness:* you have hand-written a numerical scheme and inherited the duty to validate it.

## 4. Depthwise 3×3 convolution — Interleave loses

This is the boundary case, and it is in the suite on purpose.

The plain Julia loop already reaches **40.6 GFlop/s**: LLVM auto-vectorizes along the
contiguous axis without being asked. DLI at `P=16` gives **0.43×** — that is **2.3× slower
than doing nothing**.

**What you should use instead:** [NNlib](https://fluxml.ai/NNlib.jl/stable/)'s
`depthwiseconv` for the standard primitive, `LoopVectorization.@turbo` for a custom variant,
or [Tullio.jl](https://github.com/mcabbott/Tullio.jl) in index notation. All three are
designed for exactly this shape.

**What Interleave offers here:** set `P = 1` and the same kernel returns to parity (1.05×).
That is the useful property — the escape hatch is one character, not a rewrite.

## 5. Sobel + motion — Interleave loses again

Same story: a 43.4 GFlop/s reference, `P=8` gives 0.88×, and `P=1` restores parity.

**Instead:** `LoopVectorization.@turbo`, or
[ImageFiltering.jl](https://juliaimages.org/stable/pkgs/filtering/) for standard
`imfilter`-shaped work.

These two negative cases are the reason the package documents a decision rule rather than a
speedup claim.

## 6. Per-instance reductions

Covered by `bench/reductions.jl`. A squared norm or dot product per instance has **no
recurrence**, so Base is already competitive: `sum(abs2, A; dims=2)` reduces along a
contiguous axis and vectorizes well.

DLI still has a structural argument here — the reduction is expressed once, scalar, and runs
inside the same batch traversal as the recurrence kernels that surround it, avoiding a layout
round-trip. Measure before assuming a gain.

---

## Strengths of the approach

1. **One kernel, one source.** The scalar kernel is the vectorized kernel. There is no second
   implementation to keep in sync, and the central test asserts bit-exact (`==`, not `≈`)
   agreement between the two layouts.
2. **The gain lands where compilers fail.** 12–15× on recurrences is not an incremental win;
   it is the difference between a batch being feasible and not.
3. **A one-character escape hatch.** `P = 1` returns control to the compiler and makes the
   container an ordinary dense array. A kernel that turns out not to suit DLI costs nothing
   to leave in place.
4. **Parallelism stays visible.** `apply!` cannot spawn tasks; only `parallel_apply!` does. A
   caller already inside a parallel region is never surprised.
5. **The same source reaches the GPU.** `gpu_apply!` runs the identical scalar kernel with one
   work item per problem — no packets, no second language.

## Weaknesses, stated plainly

1. **`P` must be measured.** There is no universally good default: the best value is neither
   the hardware vector width nor a constant across kernels. On this machine `P=32` beats
   `P=16` on Thomas, four times the 128-bit NEON width, because a latency-bound recurrence
   wants several vectors in flight. [`tune`](@ref) exists to answer this, but it is still a
   measurement you must make.
2. **It actively hurts already-vectorized kernels.** Not "no gain" — a real 2.3× loss if you
   leave `P>1` on a stencil. The decision rule is not optional.
3. **No domain features.** No filter design, no adaptive time-stepping, no automatic
   differentiation, no autodiff-through-the-solver. Domain libraries keep their value, and the
   right answer is often to combine them.
4. **The batch must be large and homogeneous.** `P` problems advance in lockstep. Ragged
   lengths, early exits, or data-dependent branching per instance break the model.
5. **Element types are restricted.** `Vec{P,T}` requires an LLVM leaf type, so `Vec{P,Dual}`
   does not compose: the genericity holds on each axis separately, not on their product.
6. **Threading gains are not multiplicative with `P`.** Once `P` is well chosen the kernel is
   memory-bandwidth-bound; Thomas goes from 13.4× to only 16.2× on 8 threads. Do not sell the
   two gains as a product.

## The one-line summary

Interleave is worth reaching for when the kernel is **custom, recurrence-heavy, and embedded
in a larger CPU workflow**. It loses to the compiler when the loop was already vectorizable,
and it loses to a domain library when that library already implements your exact primitive.
Measure the reference throughput first; it tells you which case you are in.
