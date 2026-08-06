# Finite-difference 16 x 16 in-vivo brain reconstruction.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("reconstruct.jl")
run_finite_difference_reconstruction((16, 16, 1))
