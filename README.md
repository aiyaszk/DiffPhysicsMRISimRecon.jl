# DiffPhysicsMRISimRecon

Differentiable MRI simulation for physics-based reconstruction.

## Node-density reconstruction

`DiffPhysicsMRISimReco.jl` creates a 126 × 126 grid of reconstruction nodes over
the measured field of view. It does not load a brain phantom or use known tissue
masks. T1 and T2 are fixed, while central finite differences estimate the 15,876
relative node densities from measured multi-coil data. KomaMRI runs the Bloch
simulations on the Apple GPU through Metal.jl. One complete gradient requires
31,752 KomaMRI simulations.

```sh
julia DiffPhysicsMRISimReco.jl
```

It saves `reconstructed_density.png` and `simulated_acquisition.mrd`.

## MRIReco comparison

Run the model-based reconstruction first, then:

```sh
julia MRIRecoResults.jl
```

The second script reconstructs the measured and node-simulated MRD files and
saves:

- `MRIRecoResults/acquisition_direct.png`
- `MRIRecoResults/acquisition_sense.png`
- `MRIRecoResults/simulated_mrd_direct.png`
- `MRIRecoResults/simulated_mrd_sense.png`

Both scripts activate and instantiate the included Julia environment.
