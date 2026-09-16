using Documenter, Interleave
using Documenter.Remotes: GitHub

# Documenter source links require at least one commit. Disable them in a new repository;
# they become active automatically after the first commit.
const REPO = let root = dirname(@__DIR__)
    cmd = `git -C $root rev-parse --verify HEAD`
    if success(pipeline(cmd; stdout = devnull, stderr = devnull))
        (; repo = GitHub("laurentplagne", "Interleave.jl"))
    else
        @info "repository has no commit: source links are disabled"
        (; remotes = nothing)
    end
end

DocMeta.setdocmeta!(Interleave, :DocTestSetup, :(using Interleave); recursive = true)

makedocs(;
    modules  = [Interleave],
    sitename = "Interleave.jl",
    authors  = "Laurent Plagne",
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://laurentplagne.github.io/Interleave.jl",
        sidebar_sitename = false,
        assets = ["assets/interleave.css"],
    ),
    pages = [
        "Why Interleave?" => "index.md",
        "Tutorials" => [
            "1 · Tridiagonal recurrence" => "tutorials/thomas.md",
            "2 · Feedback filter bank" => "tutorials/biquad.md",
            "3 · 3-D ADI sweeps" => "tutorials/adi.md",
            "4 · Decide and tune" => "tutorials/choosing-p.md",
        ],
        "Benchmark applications" => [
            "What the suite demonstrates" => "applications/index.md",
            "Thomas tridiagonal systems" => "applications/thomas.md",
            "IIR biquad filter bank" => "applications/biquad.md",
            "Black–Scholes Crank–Nicolson" => "applications/black-scholes.md",
            "Depthwise 3×3 convolution" => "applications/depthwise.md",
            "Sobel plus motion" => "applications/video.md",
            "Per-instance reductions" => "applications/reductions.md",
        ],
        "Guide" => [
            "Mental model" => "manual/concepts.md",
            "Working with a batch" => "manual/usage.md",
            "Writing kernels" => "manual/kernels.md",
            "Dimensions and packed axis" => "manual/dimensions.md",
            "Performance" => "manual/tuning.md",
            "Task parallelism" => "manual/parallel.md",
            "GPU execution" => "manual/gpu.md",
            "Positioning and alternatives" => "manual/alternatives.md",
            "What you would write instead" => "manual/what-it-replaces.md",
            "Design review and roadmap" => "manual/review.md",
        ],
        "API reference" => "reference.md",
    ],
    checkdocs = :exports,
    REPO...,
)

# `devbranch` doit être explicite. Laissé à `nothing`, Documenter tente de deviner la branche
# de développement, et le checkout superficiel d'Actions ne lui permet pas toujours d'y
# arriver : il retombe alors sur `master`, décide qu'un push sur `main` ne le concerne pas, et
# **sort sans rien déployer ni signaler d'erreur**. Le job reste vert, `gh-pages` n'est jamais
# créée, et le site annoncé par le README n'existe pas. C'est exactement ce qui s'est produit.
deploydocs(
    repo = "github.com/laurentplagne/Interleave.jl.git",
    devbranch = "main",
    push_preview = false,
)
