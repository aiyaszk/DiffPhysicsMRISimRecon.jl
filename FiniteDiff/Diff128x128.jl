# Finite-difference 128 x 128 in-vivo brain reconstruction.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("reconstruct.jl")
run_finite_difference_reconstruction((128, 128, 1))
