# DiffPhysicsMRISimRecon

Differentiable MRI simulation for physics-based reconstruction.

This research code uses multi-coil [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
as a forward model for small density and T1 reconstruction experiments. Central finite
differences are the initial gradient reference and are not intended for large images.

## Run

```sh
julia DiffPhysicsMRISimRecon.jl
```

The first run installs the dependency versions pinned in `Manifest.toml`, including the
required multi-coil KomaMRI development branch.

Use `--quick` to run only two optimizer iterations:

```sh
julia DiffPhysicsMRISimRecon.jl --quick
```

This is a standalone research project, not a Julia package. The main script runs the
experiments and `reconstruction.jl` contains the small shared helpers.
