# DiffPhysicsMRISimRecon

Differentiable MRI simulation for physics-based reconstruction.

## Voxel-density reconstruction

`DiffPhysicsMRISimReco.jl` estimates a 64 × 64 image of 4,096 relative voxel
densities at z = 0. Each voxel has 10 bilinearly interpolated in-plane spins on
each of 10 z slices, giving 100 simulation spins per voxel and 409,600 spins in
total. T1, T2, and T2* are infinite. The finite-difference step is based on the
Float32 simulation precision. Each iteration is saved under
`DiffPhysicsMRISimRecoIterations/`.

The scripts load Metal on macOS and CUDA on other systems. An optional first
argument selects the acquisition archive; otherwise `~/Desktop/Archive (1)` is
used.

```sh
julia DiffPhysicsMRISimReco.jl
```

It saves `reconstructed_density.png` and `simulated_acquisition.mrd`.

## Cross recovery test

`DiffCrossTest.jl` defines a 6 × 6 × 1 binary cross and bilinearly interpolates it onto
a 5 × 2 × 1 subspin grid (10 spins per voxel). It simulates one fully sampled EPI acquisition and
recovers the single slice with finite differences and gradient descent. It saves
the truth, initial image, every iteration, and final reconstruction under
`DiffCrossTestResults/`.

```sh
julia DiffCrossTest.jl
```

On Ultron, pass the location of the copied acquisition archive:

```sh
julia --threads=auto DiffCrossTest.jl /path/to/Archive
julia --threads=auto DiffPhysicsMRISimReco.jl /path/to/Archive
```

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
