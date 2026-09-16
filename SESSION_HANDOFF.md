# Interleave.jl — session handoff

Last updated: 2026-09-16

## Project status

Interleave.jl is the Julia port of the Data Layout Interleaving ideas from Legolas++.
The CPU path uses `Vec{P,T}` packets; the GPU path maps one independent scalar problem to
one KernelAbstractions work item. Vulkan/SPIR-V is deliberately out of the active project
and out of CI.

The GitHub repository is:

```text
https://github.com/LaurentPlagne/Interleave.jl
```

The working tree was clean after commit `e417706`:

```text
e417706 skip Metal benchmarks cleanly when runner has no GPU
```

Previous relevant commits:

- `2b455da`: CPU/Metal benchmark reset and reduction coverage;
- `9d3c296`: shared KernelAbstractions runner plus CUDA/AMDGPU environments;
- `0c5f802`: GitHub runner-name based GPU dispatch;
- `e417706`: Metal absence is reported as a skip instead of a workflow failure.

## Current benchmark matrix

| Target | Workflow job | Runner | Status |
|---|---|---|---|
| Linux CPU | `cpu-linux` | `ubuntu-latest` | runs on push/release/schedule |
| Apple CPU + Metal | `cpu-metal` | `macos-14` | CPU runs; Metal skips if the VM has no GPU |
| NVIDIA CUDA | `ka-nvidia` | configured GitHub larger runner, Tesla T4 | release or manual GPU dispatch |
| AMDGPU/ROCm | `ka-amd` | explicitly configured AMD/ROCm label | optional, not part of the requested active matrix |

The latest public workflow triggered by `e417706` was still running when this document was
written:

- benchmark workflow: [run 35078053129](https://github.com/LaurentPlagne/Interleave.jl/actions/runs/35078053129);
- package CI: [run 35078053109](https://github.com/LaurentPlagne/Interleave.jl/actions/runs/35078053109).

The preceding run `35077300442` had Linux CPU success but failed at the Metal Thomas step.
The cause was the absence of a functional Metal device on the GitHub macOS VM. The guard in
`gpu/metal/thomas.jl` and `gpu/metal/all.jl` now makes that condition a clean skip.

## How to launch the Tesla benchmark

GitHub's managed GPU larger runner is a Tesla T4, but its workflow label is the runner name
created in the repository/organization settings. It is not safe to invent a universal label.

1. In GitHub, create/enable the GPU larger runner and copy its exact label.
2. Either define the repository Actions variable:

   ```text
   INTERLEAVE_NVIDIA_RUNNER=<exact Tesla runner label>
   ```

   or provide the label as the `nvidia_runner` input when dispatching the workflow.
3. Open **Actions → Cross-platform benchmarks → Run workflow**.
4. Set `run_gpu=true` and launch it.

The CUDA environment is `gpu/cuda`, and the common benchmark entry point is:

```text
julia --project=gpu/cuda gpu/ka/all.jl
```

The workflow runs `nvidia-smi`, instantiates CUDA.jl, validates every kernel against the
scalar CPU oracle, resets resident device buffers before each timed sample, and uploads
`nvidia-cuda.txt`.

## Important files

- `.github/workflows/benchmarks.yml`: CPU, Metal, Tesla and optional ROCm jobs;
- `gpu/ka/all.jl`: backend-neutral entry point;
- `gpu/metal/all.jl`: shared KernelAbstractions implementation (`metal`, `cuda`, `amdgpu`);
- `gpu/metal/thomas.jl`: small Metal Thomas smoke benchmark;
- `gpu/cuda/Project.toml`: CUDA/Tesla environment;
- `gpu/amd/Project.toml`: AMDGPU/ROCm environment;
- `ext/InterleaveKernelAbstractionsExt.jl`: implementation of `gpu_apply!`;
- `test/runtests.jl`: CPU, packing, and KernelAbstractions CPU-backend tests;
- `docs/src/manual/gpu.md`: GPU design, runner setup, and limitations;
- `gpu/README.md`: backend environments and CI expectations.

## Validation already completed

- Julia test suite: **301 passed, 0 failed** before the last Metal-only guard change;
- local Metal smoke test of the shared runner: passed;
- workflow YAML parses successfully;
- CUDA and AMDGPU projects parse as valid Julia `Project.toml` files;
- CPU CI run for `e417706` was expected to remain unchanged; check the linked run before
  making further edits.

Julia must be run through Kaimon (`ex`/`run_tests`), not through a shell `julia` command. The
main project path is `/Users/laurentplagne/Projects/Legolas.jl`; the known Metal session was
`353498c4`.

## Next recommended actions

1. Check runs `35078053129` and `35078053109` for completion.
2. Dispatch the benchmark workflow with `run_gpu=true` and the Tesla runner label.
3. Record the Tesla output and device information in the benchmark documentation.
4. Compare CPU, Metal (when a real Apple GPU runner is available), and Tesla timings only
   after confirming the same problem sizes, synchronization policy, and transfer accounting.
5. Keep AMDGPU support dormant unless an actual ROCm runner label is provided.

Do not reintroduce Vulkan into the workflow unless a separate, maintained Julia-to-SPIR-V
execution path is deliberately adopted.
