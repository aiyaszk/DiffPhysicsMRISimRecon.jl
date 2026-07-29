# DiffPhysicsMRISimRecon

Differentiable MRI simulation for physics-based reconstruction.

## Model-based reconstruction

`DiffPhysicsMRISimReco.jl` uses CPU Bloch simulations and central finite
differences to estimate the phantom's three density levels from measured
multi-coil data. It keeps T1 fixed and saves `reconstructed_density.png`.

```sh
julia DiffPhysicsMRISimReco.jl
```

## MRIReco comparison

`MRIRecoResults.jl` reproduces the measured-versus-simulated comparison from
the KomaMRI custom-coil-sensitivity how-to.

```sh
julia MRIRecoResults.jl
```

It saves:

- `MRIRecoResults/acquisition_direct.png`
- `MRIRecoResults/acquisition_sense.png`
- `MRIRecoResults/simulated_mrd_direct.png`
- `MRIRecoResults/simulated_mrd_sense.png`

Both scripts activate and instantiate the included Julia environment.
