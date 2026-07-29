# DiffPhysicsMRISimRecon

Differentiable MRI simulation for physics-based reconstruction.

The single script `DiffPhysicsMRISimReco.jl` uses the same local 3T phantom,
accelerated EPI sequence, measured MRD data, and ESPIRiT coil maps as the KomaMRI
custom-sensitivity-map how-to. It keeps T1 fixed and reconstructs only the three
density levels present in the phantom.

## Run

```sh
julia DiffPhysicsMRISimReco.jl
```

The Koma simulations build one density basis from each tissue group. The optimizer
then minimizes the measured multi-coil data error with central finite differences.
The script activates and instantiates its own environment, runs gradient descent
directly, and saves the result as `reconstructed_density.png`.

Use ten spins per density for a fast code check:

```sh
julia DiffPhysicsMRISimReco.jl --quick
```

This is the first real-data baseline, not yet a free 128 × 128 image reconstruction.
Central differences need two forward evaluations per unknown, so a voxelwise version
would need 32,768 Koma simulations for every full gradient.
