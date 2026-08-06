# Finite-difference 64 x 64 in-vivo brain reconstruction.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("reconstruct.jl")
run_finite_difference_reconstruction((64, 64, 1))
