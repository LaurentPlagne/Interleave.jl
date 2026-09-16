# Design review and roadmap

This note records the design review of Interleave.jl and the corrective work applied
after it. It is deliberately separate from the user tutorial: it describes the current
contracts, their limits, and the evidence still required before calling the API stable.

## Assessment

The central design is sound. A scalar kernel is applied to independent instances; on the
CPU, the machine element changes from `T` to `Vec{P,T}`, while on a GPU one work item owns
one complete scalar instance. `apply!`, `parallel_apply!`, and `gpu_apply!` make the three
execution choices explicit. This imports the idea of Legolas++ without importing its C++
expression-template machinery.

The promise should nevertheless be stated precisely:

> Write the calculation of one independent instance once; adapt storage and execution to
> the target.

It should not promise that every kernel is automatically optimal on every backend. A
recurrence such as Thomas or a biquad maps naturally to one GPU work item. A 2-D stencil
needs an additional decomposition over pixels or tiles; the current one-instance-per-work-
item driver proves source reuse, but is not yet a performance-optimal stencil schedule.

## Findings fixed in this revision

* `Interleave.Array` stores a packed `data` array and an aliased scalar `flat` view. The
  generic `deepcopy` implementation copied those fields independently. Scalar indexing
  could therefore return `7` while the packed kernel still read `1`. A custom
  `Base.deepcopy_internal` now rebuilds `flat` from the copied packed storage, and the test
  suite checks the invariant.
* CPU scratch is an instance-sized prototype; GPU scratch is a batch-major device array.
  The Metal host oracle now constructs an instance-sized CPU scratch view, so it no longer
  accidentally validates a linear slice of the GPU buffer.
* All in-place CPU benchmark variants can now receive reset callbacks. Each one-evaluation
  sample starts from the same state, while reset work remains outside the timed region.
  This matters particularly for Black–Scholes, where `V` and `RHS` are overwritten.
* The shared KernelAbstractions suite resets resident device buffers before every timed
  sample. Its timings remain launch-plus-device timings, but no longer measure a different
  numerical problem on each repetition.
* The benchmark workflow explicitly selects `bash`. This enables `pipefail`, so a Julia
  failure cannot be hidden by a successful `tee` command.
* The C++ audit driver now points at the sibling Legolas++ checkout and compiles against
  its actual `Legolas::Array`/`Legolas::map` API. A small local run was used as a smoke test;
  it is still not a cross-language performance claim.

## API points to stabilize

`scratch` should eventually have an explicit preparation object or a documented distinction
between a CPU prototype and a GPU allocation. A reusable execution plan could own the
workspace, backend, workgroup size, and (for CPU) packet size, avoiding repeated setup while
keeping `apply!` allocation-free in its hot loop.

The public meaning of `instance(A, k)` also deserves clarification: for an interleaved
array `k` is a packed packet, not necessarily one logical problem. A distinct internal name
would avoid exposing this implementation detail as the public notion of an instance.

Finally, parameters such as filter coefficients should be ordinary concrete arguments or a
small callable object. Backend-specific `gpu_*` wrappers are useful as smoke-test fixtures,
but should not become a second user kernel language.

## KernelAbstractions decision

KernelAbstractions is the right portability layer for the GPU path. Its [documented
backends](https://juliagpu.github.io/KernelAbstractions.jl/stable/) include CUDA, ROCm, oneAPI,
and Metal, while backend-specific packages still provide the actual compiler and runtime.
The optional extension design keeps Interleave usable without a GPU dependency.

The CPU SIMD path should remain separate: KA provides a common GPU-like execution model, not
the best implementation of the native `Vec{P,T}` storage path. Likewise, GPU compilation is
only guaranteed for the backend-supported Julia subset (no allocation, exceptions, I/O,
tasks, or unconstrained runtime dispatch).

## Recommended next milestones

1. Finish the container audit: `copy`, `deepcopy`, serialization, views, and garbage-
   collection lifetime of the `unsafe_wrap` alias.
2. Strengthen cross-backend tests with nonuniform data, non-multiple batch sizes, boundary
   cases, and independent scratch contents. Constant images and affine Laplacians are useful
   smoke tests but are not sufficient numerical or performance tests.
3. Extend the repaired C++ comparison harness with identical problem sizes, operation
   counts, compiler flags, device information, transfers, and launch synchronization in
   every report.
4. Enable the optional CUDA and AMDGPU runner jobs and compare their results with Metal.
   This is the meaningful test of “write once, execute on several targets”.
5. Add explicit execution plans and measured tuning for `P`, CPU chunking, and GPU
   workgroup size. For stencils, introduce a separate pixel/tile execution domain rather
   than duplicating the numerical formula.

The project is therefore ready for continued development, but not yet for a broad claim of
performance portability. Its strongest current result is a credible single-source model;
the next work should make the measurements and contracts as reliable as that model.
