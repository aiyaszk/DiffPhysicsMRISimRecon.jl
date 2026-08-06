# Finite-difference 32 x 32 in-vivo brain reconstruction.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("reconstruct.jl")
run_finite_difference_reconstruction((32, 32, 1))
