"""Backend-neutral KernelAbstractions benchmark entry point.

The implementation is shared with the historical Metal path.  Select the backend with
`INTERLEAVE_KA_BACKEND=metal|cuda|amdgpu`.

    julia --project=gpu/cuda gpu/ka/all.jl
"""

include(joinpath(@__DIR__, "..", "metal", "all.jl"))

# `metal/all.jl` se termine par `abspath(PROGRAM_FILE) == @__FILE__ && main()`, qui n'est vrai
# que si c'est LUI le script lancé. Inclus depuis ici, la garde est fausse et `main()` n'était
# jamais appelé : le programme chargeait CUDA, définissait tout, et sortait sans une ligne de
# sortie ni un code d'erreur. Le job CUDA de la CI aurait téléversé un artefact vide.
main()
