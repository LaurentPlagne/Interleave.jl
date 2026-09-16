# Interleave.jl — session handoff

Last updated: 2026-09-16

## Project status

Interleave.jl is the Julia port of the Data Layout Interleaving ideas from Legolas++.
The CPU path uses `Vec{P,T}` packets; the GPU path maps one independent scalar problem to
one KernelAbstractions work item. Vulkan/SPIR-V is deliberately out of the active project
and out of CI.

Repository: <https://github.com/LaurentPlagne/Interleave.jl>. The local working directory is
still named `Legolas.jl`; only that directory keeps the old name.

Julia must be run through Kaimon (`ex` / `run_tests`), not through a shell `julia` command.

## ⚠️ Correction to the previous handoff

The previous revision of this file instructed the next session to "create the GPU larger
runner and copy its exact label", then dispatch the benchmark workflow against a Tesla T4.
**That task cannot be performed.** GitHub-hosted larger runners, GPU ones included, require an
organization on a Team or Enterprise Cloud plan; `LaurentPlagne/Interleave.jl` is a personal
account. There is no runner to create and no label to copy.

The belief that Legolas++ had obtained NVIDIA numbers from GitHub was also wrong. Legolas++'s
`.github/` contains zero references to NVIDIA or CUDA: its only benchmark runners are
`ubuntu-latest` and `macos-14`. Its published NVIDIA figures (`../Legolas/vulkan.md`) come
from a workstation with a Ryzen 5 3600 and a GeForce RTX 2060 SUPER, run by hand and pasted
into the documentation.

Do not re-open this. The workflow comments, `gpu/README.md`, `README.md`, and
`docs/src/manual/gpu.md` now all state the constraint explicitly.

## How to obtain NVIDIA numbers

`gpu/run_remote.sh` is the supported path, and works unchanged on a rented box (RunPod,
Vast.ai, Lambda), a Colab runtime, or an institutional workstation:

```bash
git clone https://github.com/LaurentPlagne/Interleave.jl && cd Interleave.jl
./gpu/run_remote.sh cuda      # or amdgpu / metal; omit to guess from the hardware
```

It installs Julia through juliaup if absent, checks for Julia ≥ 1.11, records the device,
instantiates `gpu/cuda`, and runs the suite. Only the vendor *driver* is needed: CUDA.jl ships
its own toolkit as Julia artifacts, so there is no CUDA SDK to install.

Two alternatives, documented in `docs/src/manual/gpu.md`: register a self-hosted runner and
set `INTERLEAVE_NVIDIA_RUNNER` to its label (the `ka-nvidia` job then works as written), or
apply for JuliaGPU Buildkite access via the JuliaLang Slack `#gpu` channel.

The scientifically strongest measurement would be on an RTX 2060 SUPER, which makes the result
directly comparable with the Legolas++ Vulkan table in `../Legolas/vulkan.md` on identical
hardware. Vast.ai still lists that generation of card.

## Current benchmark matrix

| Target | Workflow job | Runner | Status |
|---|---|---|---|
| Linux CPU | `cpu-linux` | `ubuntu-latest` | runs on push/release/schedule |
| Apple CPU + Metal | `cpu-metal` | `macos-14` | CPU runs; Metal skips if the VM has no GPU |
| NVIDIA CUDA | `ka-nvidia` | self-hosted label only | dormant; no managed runner is available |
| AMDGPU/ROCm | `ka-amd` | self-hosted label only | dormant |

## Work completed in this session

**Three silent-correctness bugs**, none of which showed a symptom:

- `Serialization` reconstructed `Interleave.Array` field by field, so a round-trip returned
  `data` and `flat` as independent buffers and later scalar writes landed where no kernel
  reads. Fixed by `ext/InterleaveSerializationExt.jl`.
- `permutedims!` used `perm` where Base's convention is `invperm(perm)`. Right by accident on
  transpositions, wrong on 3-cycles — which a closed ADI cycle necessarily contains.
- `gpu/ka/all.jl` never called `main()`: the include-guard in `gpu/metal/all.jl` is false when
  included rather than run. The CUDA CI job would have uploaded an empty artifact and reported
  success. Found by running the documented command on Colab and getting silence.

**Four API changes**, all breaking and all made now because the package is unregistered:

- `packet(A, k)` is the hot path; `instance(A, b)` now always means problem `b`. They used to
  be one function whose meaning differed by a factor of `P`.
- `scratchlike(A)` replaces `similar(instance(A, 1))`; `gpu_scratchlike` is deliberately a
  separate name because the shapes are not interchangeable.
- The drivers diagnose contract violations. Branching on data used to give
  `non-boolean (Vec{8,Bool})` with no way out — and `Base.ifelse` has no `Vec` method either.
  The answer is `vifelse`, now re-exported, which also works on scalars so the kernel stays
  valid at `P = 1`.
- `tune(f, make)` measures `P` instead of guessing it. It reproduces the documented table and
  finds `P = 32` at 18.3× on Thomas, past the last column that table recorded.

**ADI repacking.** `permutedims!` is specialised for interleaved batches: 34% off a full cycle,
with repacking falling from 47% of the cycle to 20%. The three transitions differ by 3.7×, and
parity forces at least one 3-cycle per cycle — no layout choice avoids it.

**Both GPU backends measured** at the same commit. The T4 wins 2.4–5.9× on recurrences and
loses 1.2–1.9× on stencils, which is the driver's known limitation rather than the hardware's.
More important: **Metal is bit-exact against the CPU oracle on all ten kernels and CUDA is not**
on nine, almost certainly FMA contraction. The CPU guarantee is unchanged; the GPU one covers
the algorithm, not the rounding. Documented in `docs/src/manual/gpu.md`.

**Comparative benchmarks**, all in `bench/` behind `bench/studies.jl` (90 s at reduced sizes,
not in CI): DSP.jl loses by 27× on a filter bank while winning on features; hand-written SoA
beats DLI at recurrence order 1 and loses by 8× at order 16; adding a recursive filter flips
the Sobel pipeline from a 0.69× loss to a 3.6× win.

**Documentation**: LoopVectorization withdrawn from ten recommendation sites — it is
maintenance-only and falls back to `@inbounds @fastmath` on Julia ≥ 1.11, which is precisely
what invariant 1 forbids. The audio demo is regenerated by this package's own kernel rather
than copied. The front page states two decision criteria instead of one.

Test suite: **406 passed, 0 failed** (was 301 at the start of the session).

### A recurring methodological failure, recorded because it repeated

Three times a measurement was made against a strawman and had to be redone:

- the video benchmark compared against a batch-major reference instead of the contiguous
  layout the documented benchmark uses, turning a real loss into a fake 5× win;
- `permutedims!` was compared against a hand-written fallback in this package, which was both
  slower than Base's and wrong, inflating a 34% gain into a claimed 57%;
- the audio demo's first verification used a one-pole high-pass that let the untouched
  fundamentals through and hid an inaudible filter.

Each time the error was caught by a question rather than by the test suite. **The reference in
any comparison must be the strongest thing a user would actually write**, and a fallback
counts as a reference.

## Manual steps that need the repository owner

1. **GitHub Pages**: enable it (*Settings → Pages*, source `gh-pages`). For a reliable build,
   generate a deploy key with `DocumenterTools.genkeys()` and add it as the `DOCUMENTER_KEY`
   secret — a `gh-pages` push made with `GITHUB_TOKEN` does not trigger the Pages job.
2. **Codecov**: add the `CODECOV_TOKEN` secret. The upload is configured with
   `fail_ci_if_error: false`, so CI stays green until it exists.
3. ~~NVIDIA numbers~~ **Done.** Tesla T4 on Colab, driver 580.82.07, CUDA 13.0. The table and
   both device blocks are in `docs/src/manual/gpu.md`.
4. `gh` is not installed locally, so no session can check workflow run status until
   `brew install gh && gh auth login`.

## Next recommended actions

1. Register the package. The name `Interleave` is free in General (checked 2026-09-16), and
   `AGENTS.md` §6 no longer blocks it.
2. Settle the two API points that `docs/src/manual/review.md` flags before a public release:
   `instance(A, k)` exposes a *packet* index under the name of a logical instance, and `scratch`
   means an instance-sized prototype on CPU but a batch-major device array on GPU — an asymmetry
   that has already produced one bug in the Metal oracle.
3. Extend the container audit to JLD2/BSON/Arrow, or document that they are unsupported. The
   `data`/`flat` alias breaks under any field-wise reconstruction; two are now handled, the rest
   are not.
4. Compare CPU, Metal, and NVIDIA timings only after confirming identical problem sizes,
   synchronization policy, and transfer accounting.

Do not reintroduce Vulkan into the workflow unless a separate, maintained Julia-to-SPIR-V
execution path is deliberately adopted.
