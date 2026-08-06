# Finite-difference 8 x 8 in-vivo brain reconstruction.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("reconstruct.jl")
run_finite_difference_reconstruction((8, 8, 1))
