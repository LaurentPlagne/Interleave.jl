# Prepare the documentation environment: Documenter and the local package under development.
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
Pkg.instantiate()
