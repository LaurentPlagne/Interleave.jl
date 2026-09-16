#!/usr/bin/env bash
#
# Run the KernelAbstractions benchmark suite on a throwaway GPU machine.
#
# The point of this script is that the procedure should not live in a chat log or in a
# maintainer's memory. It works unchanged on a rented box (RunPod, Vast.ai, Lambda), on a
# Colab runtime, and on an institutional workstation. It installs Julia if needed, resolves
# the GPU environment, records the device, and prints a report that can be pasted into the
# documentation as-is.
#
#   ./gpu/run_remote.sh              # backend guessed from the hardware present
#   ./gpu/run_remote.sh cuda         # force a backend: cuda | amdgpu | metal
#
# CUDA.jl and AMDGPU.jl ship their own toolchains as Julia artifacts, so only the vendor
# *driver* has to be present. There is no CUDA toolkit or ROCm SDK to install by hand.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------------
# 1. Backend selection

BACKEND="${1:-}"
if [[ -z "$BACKEND" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        BACKEND=cuda
    elif command -v rocminfo >/dev/null 2>&1; then
        BACKEND=amdgpu
    elif [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
        BACKEND=metal
    else
        echo "error: no GPU detected; pass a backend explicitly (cuda | amdgpu | metal)" >&2
        exit 1
    fi
    echo "==> detected backend: $BACKEND"
fi

case "$BACKEND" in
    cuda)   ENV_DIR=gpu/cuda  ;;
    amdgpu) ENV_DIR=gpu/amd   ;;
    metal)  ENV_DIR=gpu/metal ;;
    *) echo "error: unsupported backend '$BACKEND' (expected cuda, amdgpu, or metal)" >&2
       exit 1 ;;
esac

# ---------------------------------------------------------------------------------
# 2. Julia
#
# The GPU environments use a `[sources]` entry to reach the unregistered parent package,
# which Pkg only understands from Julia 1.11 onwards.

if ! command -v julia >/dev/null 2>&1; then
    echo "==> installing Julia via juliaup"
    curl -fsSL https://install.julialang.org | sh -s -- --yes
    export PATH="$HOME/.juliaup/bin:$PATH"
fi

julia --version
julia -e 'VERSION >= v"1.11" || error("Julia >= 1.11 is required: the GPU environments use [sources]")'

# ---------------------------------------------------------------------------------
# 3. Device identification
#
# A timing without the device it ran on is not a result. Capture this next to the numbers.

echo
echo "===================== device ====================="
case "$BACKEND" in
    cuda)   nvidia-smi || true ;;
    amdgpu) rocminfo 2>/dev/null | head -40 || true; rocm-smi || true ;;
    metal)  system_profiler SPDisplaysDataType 2>/dev/null | head -20 || true ;;
esac
echo "=================================================="
echo

# ---------------------------------------------------------------------------------
# 4. Resolve and run
#
# The first instantiate downloads the vendor artifacts (1-2 GB for CUDA); expect a few
# minutes on a fresh machine.

echo "==> instantiating $ENV_DIR"
julia --project="$ENV_DIR" -e 'using Pkg; Pkg.instantiate()'

echo "==> running the KernelAbstractions suite on $BACKEND"
INTERLEAVE_KA_BACKEND="$BACKEND" julia --project="$ENV_DIR" gpu/ka/all.jl

echo
echo "==> done. Paste the device block and the table above into docs/src/manual/gpu.md."
