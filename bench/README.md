# Benchmarks

Two entry points, answering two different questions.

## `runall.jl` — what is DLI worth on this kernel?

```bash
julia --project=bench -t auto bench/runall.jl
```

One section per shipped kernel (Thomas, biquad, depthwise, video, option pricing,
reductions), each sweeping the packet size `P` against a naive reference. This is what the
`Cross-platform benchmarks` workflow runs on Linux and Apple Silicon, so it must stay within
a 30-minute budget on a 2-core runner.

Method, per [`harness.jl`](harness.jl): interleaved A/B — every variant is played on every
round and the minimum per variant is kept, which cancels slow machine-load drift. Load average
is recorded before measuring, and any vectorization claim is corroborated structurally by
looking for `<P x float>` in the emitted LLVM.

## `studies.jl` — what would the alternative cost?

```bash
julia --project=bench bench/studies.jl        # quick tour, ~90 s, modest sizes
```

Comparative studies rather than per-kernel numbers. **Not run by CI**: the published sizes
allocate several GB. The quick tour above uses reduced sizes and reproduces every qualitative
conclusion; the tables in
[`docs/src/manual/what-it-replaces.md`](../docs/src/manual/what-it-replaces.md) come from each
file's **own defaults**, reproduced by running it directly:

| file | question | published size | note |
|---|---|---|---|
| [`dspcompare.jl`](dspcompare.jl) | Interleave against DSP.jl on a filter bank | 1024 × 4096 | each library in its natural layout |
| [`soavsdli.jl`](soavsdli.jl) | tridiagonal: DLI against hand-written global SoA | up to 2 097 152 | ~3 GB |
| [`pentasoa.jl`](pentasoa.jl) | pentadiagonal: same, with the scratch doubled | up to 2 097 152 | **~5 GB** |
| [`orderscan.jl`](orderscan.jl) | order-`M` IIR: does the gap track the predecessor count? | 65 536 × 1024 | ~1 GB |
| [`videorec.jl`](videorec.jl) | video: does a recurrence flip the verdict? | 256 × 128² | against contiguous images |

```bash
julia --project=bench bench/pentasoa.jl       # reproduces the published table
```

Every study checks numerical agreement **before** timing, and requires the DLI variant to be
bit-exact against the scalar reference — `==`, not `≈`, per invariant 1 in
[`AGENTS.md`](../AGENTS.md). A study whose reference comes out identically zero, or whose DLI
result differs by one ulp, fails rather than reporting a meaningless speedup.

## Adding a study

Give it a `run(; kwargs...)` with size keywords, so `studies.jl` can call it small and a
maintainer can call it at the published size. Validate before timing. Keep the reference
*strong*: comparing against a deliberately bad layout produces a number that flatters the
library and tells the reader nothing. One earlier draft of `videorec.jl` did exactly that and
had to be rewritten.
