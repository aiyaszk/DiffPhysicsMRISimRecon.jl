# DifferentiableKomaMRI.jl

Experimental nonlinear inverse problems using multi-coil
[KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl) as the forward model.

The initial baseline uses central finite differences to calculate gradients and compares
gradient descent with L-BFGS for:

- fully sampled image reconstruction;
- genuinely accelerated `R=2` EPI acquisition; and
- joint proton-density and T1 reconstruction from multiple inversion times.

Finite differences are intentionally the reference implementation. They require two
forward simulations per parameter and will not scale to clinical image sizes. Their role
is to validate objectives and future automatic-differentiation or adjoint gradients.

## Development setup

The current examples require the multi-coil KomaMRI development branch:

```julia
using Pkg
Pkg.add(url="https://github.com/aiyaszk/KomaMRI.jl", rev="multi-coils-blochsimple-support")
Pkg.develop(path="/path/to/DifferentiableKomaMRI.jl")
Pkg.test("DifferentiableKomaMRI")
```

## Examples

```sh
julia --project=. examples/01_density_reconstruction.jl
julia --project=. examples/02_t1_reconstruction.jl
```

Pass `--quick` during development to limit each optimizer to two iterations.

## Roadmap

1. Establish finite-difference gradient and optimizer baselines.
2. Add noise, regularization, and measured sensitivity maps.
3. Scale the full and accelerated experiments beyond the 4-by-4 validation problem.
4. Implement matrix-free sensitivity or adjoint gradients and compare them against finite
   differences.
