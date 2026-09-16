# What you would write instead

[Positioning and alternatives](alternatives.md) surveys the tools. This page answers the
narrower and more useful question, kernel by kernel: **if Interleave did not exist, what would
you actually write, and would you be worse off?**

Two of the six answers are "you would be better off". Those are the important ones. Where a
competing package exists, it is **measured** rather than characterised — see the DSP.jl table
below, which corrects an earlier unmeasured claim on this page.

Every table here is reproduced by a file in [`bench/`](https://github.com/LaurentPlagne/Interleave.jl/tree/main/bench).
`julia --project=bench bench/studies.jl` runs the whole set at reduced sizes in about 90
seconds and reaches the same conclusions; each file's own defaults reproduce the exact numbers
printed here.

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

!!! tip "Two questions decide, and neither is the application domain"
    **1. What throughput does your reference reach?** At 1–3 GFlop/s the compiler has failed —
    there is a recurrence, and DLI returns 12–15×. At 40+ GFlop/s LLVM already vectorized the
    inner loop; DLI can then only *take over* that gain, never add to it, and it takes it over
    badly.

    **2. How many predecessors does the recurrence carry?** This decides how DLI compares to
    the serious alternative, a hand-written SoA layout. One predecessor and SoA wins; beyond
    that DLI pulls ahead and keeps pulling — 2.2× on a tridiagonal solve, 11.8× on a
    pentadiagonal one, 8.4× on a 16th-order IIR. The sweeps are below.

    The two questions are independent, and the second can overturn the first: adding a
    recursive filter to the Sobel pipeline — a documented *loss* at 0.69× — turns it into a
    3.6× win without touching the stencil.

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
rewrite the kernel as "time outside, instances inside", with `@simd` on the instance loop.
That is legal — instances are independent — and it is DLI done by hand. It is measured below.

**Verdict.** 13.4× at `P=16`, and [`tune`](@ref) finds 18.3× at `P=32` on this machine — the
packet size is genuinely worth measuring rather than assuming.

*Strength:* the kernel source is unchanged and stays bit-exact against its scalar self.
*Weakness:* you own a tuning parameter you did not have before.

### DLI against a hand-written global SoA

Transposing to global SoA is the serious alternative, so it deserves a measurement rather
than an assertion. `bench/soavsdli.jl` runs three traversals over **the same**
`(nbatch, nx)` array — identical footprint, only the loop order differs:

- *reference*: `b` outside, `i` inside — strided along the recurrence;
- *SoA*: `i` outside, `b` inside with `@simd` — instances contiguous;
- *DLI*: one packet crosses the whole recurrence before the next starts.

All three agree bit-exactly. Thomas, `nx = 64`, `P = 16`, `Float32`, one core of an Apple
M1 Max (L1d 128 KB, **L2 12 MB**):

| `nbatch` | SoA working set per `i` step | reference | SoA `@simd` | DLI | DLI / SoA |
|---:|---:|---:|---:|---:|---:|
| 256 | 7 KB | 0.15 ms | 0.01 ms | 0.01 ms | 1.38× |
| 1 024 | 28 KB | 0.83 ms | 0.08 ms | 0.03 ms | 2.26× |
| 16 384 | 448 KB | 25.60 ms | 1.08 ms | 0.55 ms | 1.98× |
| 262 144 | 7.0 MB | 700.84 ms | 20.24 ms | 9.87 ms | 2.05× |
| 524 288 | 14.0 MB | 1065.75 ms | 43.61 ms | 18.82 ms | 2.32× |
| 1 048 576 | 28.0 MB | 2369.29 ms | 80.81 ms | 37.44 ms | 2.16× |
| 2 097 152 | 56.0 MB | 6063.74 ms | 165.68 ms | 74.36 ms | 2.23× |

**DLI beats hand-written SoA by a steady factor of about 2.2 on this kernel, and the ratio
does not grow with the population.** The sweep deliberately crosses the 12 MB L2 boundary —
from 7 KB to 56 MB, 4.6× past it — with no visible change.

On *this* kernel the mechanism is memory traffic rather than cache behaviour. Both variants are
bandwidth-bound and reach comparable bandwidth (36 versus 29 GB/s single-threaded); SoA simply
moves about 1.8× more bytes, because back substitution needs one scratch value per instance per
step, so the SoA formulation must materialise an entire `nbatch × nx` workspace in DRAM while
the DLI workspace is `nx × P` and never leaves L1.

### The tridiagonal result does not generalise

It would be easy to stop there and conclude that the DLI advantage is a modest constant. That
conclusion is wrong, and `bench/pentasoa.jl` shows why. A **pentadiagonal** solve needs *two*
auxiliary sequences carried to the back substitution instead of one, so the SoA form
materialises `2 × nbatch × n` of workspace and reaches back *two* columns in several arrays at
once. Same three traversals, same bit-exact agreement, `n = 64`, `P = 16`:

| `nbatch` | SoA scratch | reference | SoA `@simd` | DLI | DLI / SoA |
|---:|---:|---:|---:|---:|---:|
| 16 384 | 8 MB | 34.96 ms | 10.46 ms | 1.14 ms | **9.19×** |
| 262 144 | 128 MB | 886.40 ms | 200.16 ms | 17.73 ms | **11.29×** |
| 1 048 576 | 512 MB | 4046.66 ms | 870.45 ms | 76.77 ms | **11.34×** |
| 2 097 152 | 1024 MB | 8186.06 ms | 1734.67 ms | 147.49 ms | **11.76×** |

The gap goes from 2.2× to nearly **12×**, and a traffic-only model does not explain it —
traffic differs by just 1.57×. The achieved bandwidth is where it happens:

| pentadiagonal, `nbatch` = 2 097 152 | bytes moved | achieved |
|---|---:|---:|
| DLI | 3.76 GB | 25.5 GB/s |
| SoA | 5.91 GB | **3.4 GB/s** |

**SoA's achieved bandwidth collapses by 7.5×**, and 1.57 × 7.5 ≈ 11.8 accounts for the
measurement. The cause is the number of simultaneous batch-major streams: roughly 7 for the
tridiagonal form, 11 to 13 for the pentadiagonal one, each striding `nbatch × 4` bytes — 8 MB
at this size — between consecutive `i`. That exceeds what the hardware prefetchers and the TLB
sustain, and the memory system falls off a cliff. DLI is immune by construction: its entire
per-packet working set is `n × P` per array, about 37 KB in total here, **whatever `nbatch` is
and however many bands the matrix has**.

So the locality argument for AoSoA is real after all — it simply needs enough concurrent
streams to appear. The tridiagonal kernel sits below that threshold and shows only the traffic
effect; the pentadiagonal kernel sits above it. The honest summary is that **the DLI advantage
over hand-written SoA grows with the number of arrays the recurrence must carry**, not with the
size of the population.

### It is the number of predecessors that decides

Tridiagonal gave 2.2×, pentadiagonal 11.8×. Those are two points on a curve, and the curve has
a parameter: **how many predecessors the recurrence carries**. `bench/orderscan.jl` makes it
continuous with an order-`M` IIR,

```math
y[n] = b_0 x[n] + \sum_{k=1}^{M} b_k x[n-k] - \sum_{k=1}^{M} a_k y[n-k],
```

where the SoA form must keep `2M+1` concurrent batch-major streams while DLI holds `2M` state
values in registers. 65 536 channels × 1024 samples, `P = 16`, all three variants bit-exact:

| `M` | SoA streams | reference | SoA `@simd` | DLI | DLI / SoA |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 4393 ms | 12.74 ms | 31.51 ms | **0.40×** |
| 2 | 5 | 5148 ms | 229.29 ms | 43.96 ms | 5.22× |
| 4 | 9 | 793 ms | 342.34 ms | 79.01 ms | 4.33× |
| 8 | 17 | 2436 ms | 1165.68 ms | 146.71 ms | 7.95× |
| 16 | 33 | 7245 ms | 2700.83 ms | 321.75 ms | **8.39×** |

Reproducible to about 2% across runs. Two things to take from it.

**There is a floor, and it matters.** At `M = 1` — three streams — hand-written SoA *beats*
DLI by 2.5×. A first-order recurrence is the ideal case for the SoA form: few streams,
perfectly vectorized, no workspace. The advantage of DLI is not universal in the order either.

**Above that floor the advantage grows with `M`**, from 5.2× to 8.4×, with a dip at `M = 4`
that breaks strict monotonicity. The reference column is reproducibly non-monotonic too and is
not the object of this study; it is reported rather than explained.

### The same effect on a 2-D kernel, where it flips the verdict

The Sobel-plus-motion pipeline is the documented *loss*: a pure stencil, already vectorized.
`bench/videorec.jl` prepends what a real image pipeline does — a **recursive** smoothing along
rows, Deriche/van Vliet family, of order `M` — and keeps the Sobel. `M = 0` is the original
kernel. The reference is the strong one: `(H, W, nstream)` arrays where each image is
contiguous, exactly as `bench/video.jl` uses. 256 streams of 128×128, `P = 8`, bit-exact:

| `M` | reference | DLI | DLI / reference |
|---:|---:|---:|---:|
| 0 | 4.33 ms | 6.29 ms | **0.69×** |
| 1 | 8.40 ms | 6.27 ms | **1.34×** |
| 2 | 12.70 ms | 6.82 ms | 1.86× |
| 4 | 23.66 ms | 7.67 ms | 3.09× |
| 8 | 39.59 ms | 11.05 ms | **3.58×** |

**The verdict flips between `M = 0` and `M = 1`** and then grows. The mechanism is visible in
the columns rather than the ratio: DLI's time rises by 1.8× across the sweep while the
reference's rises by 9.1×. DLI does not get faster — *the alternatives get slower*, because a
row-scan recurrence is exactly what the compiler cannot vectorize, and DLI absorbs it into
registers.

(`M = 0` reads 0.69× here against 0.88× in the summary table: this kernel writes through an
extra smoothing buffer that the original does not, which costs both variants equally but is
not the same measurement.)

### And the SoA source is genuinely harder to get right

This is usually argued as a matter of taste. It is not only that. Writing these benchmarks,
the DLI variant was in every case the *scalar kernel, unchanged* — that is the whole premise,
and it cannot drift from the reference because it **is** the reference.

Each SoA variant, by contrast, needed a second implementation: a transposed loop nest, an
explicit boot phase for the first `M` samples, and separate batch-sized workspaces. The boot
phase is where a real bug appeared here — it accumulated `acc + (b·x - a·y)` where the
generated steady-state expression computes `(acc + b·x) - a·y`. Different associativity, one
ulp in `Float32`, and bit-exactness with the scalar reference was silently lost. It looked like
a property of SoA until it was tracked down; it was a typo in a loop that exists only because
the layout changed.

That is the ergonomic argument made concrete: **the alternative is a second source of truth,
and second sources drift.**

!!! note "Four kernels, one machine"
    Tridiagonal, pentadiagonal, an order-`M` IIR and a recursive video filter, on an M1 Max. The two differ by more than 5×,
    which is the point: a kernel whose SoA form needs no batch-sized workspace would narrow the
    gap further still, and a machine with different prefetcher or TLB limits would move the
    threshold. Re-run `bench/soavsdli.jl` and `bench/pentasoa.jl` rather than porting a
    verdict.

## 2. IIR biquad filter bank — measured against DSP.jl

**Without Interleave** you would use [DSP.jl](https://docs.juliadsp.org/stable/filters/) and
`filt`. `filt(b, a, X)` on a matrix filters each column independently, so a filter bank is one
call — it is the direct competitor, not an approximation of one.

An earlier revision of this page asserted that DSP.jl was "a real, well-optimized answer"
without measuring it. It is worth separating the two claims, because only one survives.

`bench/dspcompare.jl`, 1024 independent channels × 4096 samples, `Float32`, same coefficients,
each library in its own natural layout (DSP filters columns of `(ns, nb)`, Interleave works
batch-major `(nb, ns)`), numerical agreement checked before timing:

| variant | time | vs naive loop | vs `DSP.filt!` |
|---|---:|---:|---:|
| naive Julia loop | 27.321 ms | 1.00× | 0.82× |
| `DSP.filt` | 22.545 ms | 1.21× | 0.99× |
| `DSP.filt!` (in place) | 22.334 ms | 1.22× | 1.00× |
| Interleave `P=1` | 13.001 ms | 2.10× | 1.72× |
| Interleave `P=8` | 1.622 ms | 16.84× | 13.77× |
| **Interleave `P=16`** | **0.820 ms** | **33.31×** | **27.23×** |
| Interleave `P=32` | 1.070 ms | 25.54× | 20.88× |

**On throughput, DSP.jl loses by a factor of 27.** That is not a criticism of DSP.jl: `filt`
runs the same strict temporal recurrence per channel that no compiler vectorizes, so it beats
a naive loop by only 1.22×. The recurrence is the wall, and DSP.jl does not try to go around
it by batching across channels.

Two details in that table are worth more than the headline:

- **`P=1` already wins 2.10×** with no SIMD at all. An interleaved instance is contiguous,
  while `X[c, n]` in the naive column-major loop strides across channels. Part of the gain is
  pure locality, not vectorization.
- **`P=32` is *worse* than `P=16` here**, where the opposite is true for Thomas. The optimum
  is genuinely per-kernel, which is why [`tune`](@ref) exists.

Interleave computes bit-exactly the same result as the naive loop (max difference exactly
`0.0`); DSP.jl differs by `2.4e-7`, having its own accumulation order.

**Verdict.** DSP.jl wins decisively on **features** — filter design (Butterworth, Chebyshev),
form conversion, stability analysis — none of which Interleave has or should have. It loses on
**throughput for a bank** by 27×. The honest recommendation is unchanged in shape but sharper
in reason: **design the filter with DSP.jl, run the bank with Interleave.**

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

### The same algorithm wins in C++ — and that is not a contradiction

Legolas++, the C++ project these ideas come from, measured *the same* Sobel-plus-temporal
pipeline and found DLI **winning**:

| Sobel + temporal, 32×720p (C++, AVX2, Ryzen 5 3600, GCC 15.2 `-O3 -march=native`) | time | vs scalar |
|---|---:|---:|
| CPU scalar, 1 core | 56.17 ms | 1.00× |
| **CPU DLI AVX2, 1 core** | 19.95 ms | **2.82×** |

2.82× there, 0.88× here, on the same algorithm. The explanation is not that DLI behaves
differently — it is that **the two verdicts are measured against different baselines**:

| | baseline throughput | DLI verdict |
|---|---:|---:|
| C++ video pipeline | 12.1 GFlop/s | **2.82×** |
| C++ depthwise | 49.2 GFlop/s | 1.09× |
| Julia Sobel + motion | 43.4 GFlop/s | 0.88× |
| Julia depthwise | 40.6 GFlop/s | 0.43× (`P=1`: 1.05×) |

Read the first column and the second follows. GCC did **not** vectorize the C++ video kernel —
12 GFlop/s is a scalar baseline — so DLI recovered what the compiler had left on the table.
GCC *did* vectorize the C++ depthwise kernel, and DLI gained nothing there either (1.09×).
LLVM vectorized **both** Julia kernels, so DLI has nothing left to recover in either.

So the decision rule does not merely survive the cross-language comparison; it **predicts**
it. What changes between the two projects is not the technique but which kernels the compiler
happened to handle.

The practical consequence is a warning: **a DLI verdict is not portable.** It is a property of
your kernel, your compiler and your machine together, and a 2.82× measured in C++ with GCC on
AVX2 tells you nothing about the same algorithm in Julia with LLVM on NEON. Measure the
reference throughput on the machine you will actually run on.

!!! note "This is a comparison of published numbers, not a controlled experiment"
    The two campaigns differ in machine (Ryzen 5 3600 / AVX2 versus Apple M-series / NEON),
    in problem size (32×720p ≈ 29.5 Mpixels and 118 MB per buffer, against 256×128² ≈ 4.2
    Mpixels and about 16 MB here), and therefore in memory regime. The baseline throughputs
    are directly comparable; the speedups should be read as evidence for the mechanism, not
    as a head-to-head between the two implementations.

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
