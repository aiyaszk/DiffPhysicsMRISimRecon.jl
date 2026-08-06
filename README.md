# DiffPhysicsMRISimRecon

Physics-based MRI reconstruction with reverse-mode automatic differentiation and finite differences.

## Layout

- `AD/`: shared AD reconstruction and resolution entry points.
- `FiniteDiff/`: shared finite-difference reconstruction and resolution entry points.
- `MRIRecoResults/`: measured-versus-simulated MRIReco comparisons.
- `DiffCrossAccTestResults/`: accelerated cross-test results.
- `DiffCrossTestResults/`: fully sampled cross-test results.

The protected result folders remain at the repository root. Resolution-specific brain results are stored beside their method, for example `AD/ADDiff8x8/` and `FiniteDiff/Diff8x8/`.

## Brain reconstruction

Run either method from the repository root:

```sh
julia --project=. AD/ADDiff8x8.jl
julia --project=. FiniteDiff/Diff8x8.jl
```

Replace `8x8` with `16x16`, `32x32`, `64x64`, or `128x128`. An optional first argument selects the acquisition archive; otherwise `~/Desktop/Archive (1)` is used.

## Finite-difference checks

```sh
julia --project=. FiniteDiff/DiffCrossTest.jl
julia --project=. FiniteDiff/DiffCrossAccTest.jl
julia --project=. FiniteDiff/MRIRecoResults.jl
```
