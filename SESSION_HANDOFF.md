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

- **Fixed a confirmed silent-corruption bug.** `Serialization` reconstructs a struct field by
  field, so an `Interleave.Array` round-trip returned `data` and `flat` as two independent
  buffers: subsequent scalar writes landed where no kernel reads. `deepcopy` had been fixed
  earlier; serialization had not. New extension `ext/InterleaveSerializationExt.jl`, plus a
  regression test covering values, padding, the alias itself, and a kernel run on the
  deserialized batch.
- **Strengthened the GPU validation data.** The shared suite and the CPU-backend GPU testsets
  used constant inputs, which made several comparisons vacuous — a Sobel filter of a constant
  image is zero everywhere, and a constant image is invariant under transposition, so an `i`/`j`
  index swap would have passed. All inputs now vary along every axis including the batch, every
  case asserts its reference is not identically zero, and `H ≠ W` so shape errors surface.
- **Exercised the masking path on the GPU suite.** Every batch size in `gpu/metal/all.jl` was a
  multiple of the 256 work-group width, so the `batch <= nbatch` guard was never tested. Sizes
  are now 4093, 1021, 251, 1019, 127, 253.
- **Fixed the GPU environment compat.** The four `gpu/*/Project.toml` use a `[sources]` entry,
  which Pkg only understands from Julia 1.11, while declaring `julia = "1.10"`. On a fresh 1.10
  the unregistered parent package would not resolve.
- **CI: documentation is now actually deployed.** `docs/make.jl` called `deploydocs`, but the
  `docs` job had no write permission and no token, so the deployment silently did nothing while
  the README advertised the site. Added `permissions: contents: write`, the token env, and a
  `tags: ['*']` trigger for versioned docs.
- **CI: coverage is now actually uploaded.** `Pkg.test(; coverage=true)` collected data that
  never left the runner, despite `.codecov.yml` existing. Added processing and upload on the
  Linux/1.11 cell only.
- **Synchronized `AGENTS.md`**, whose title, invariants, and "Points ouverts" still referred to
  a package named `Legolas` and claimed registration was impossible under that name.

Test suite after these changes: **315 passed, 0 failed** (was 301).

## Manual steps that need the repository owner

1. **GitHub Pages**: enable it (*Settings → Pages*, source `gh-pages`). For a reliable build,
   generate a deploy key with `DocumenterTools.genkeys()` and add it as the `DOCUMENTER_KEY`
   secret — a `gh-pages` push made with `GITHUB_TOKEN` does not trigger the Pages job.
2. **Codecov**: add the `CODECOV_TOKEN` secret. The upload is configured with
   `fail_ci_if_error: false`, so CI stays green until it exists.
3. **NVIDIA numbers**: run `./gpu/run_remote.sh cuda` on a rented or borrowed GPU machine and
   paste the device block and table into `docs/src/manual/gpu.md`.
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
