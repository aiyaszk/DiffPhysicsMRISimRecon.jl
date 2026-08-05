# Reconstructs an 8 × 8 brain density from measured accelerated multi-coil data.

using Pkg # Uses the repository's Julia environment.
Pkg.activate(@__DIR__) # Selects the environment beside this script.
Pkg.instantiate() # Installs any missing declared dependencies.

using FiniteDiff, KomaMRI, MRICoilSensitivities # Provides gradients, simulation, and ESPIRiT.
using Interpolations: Flat, linear_interpolation # Maps voxel densities to simulation spins.
using KomaMRIBase: ArbitraryCoilSens, Phantom # Builds the measured-coil scanner and object.
using KomaMRICore: ISMRMRD_ACQ_IS_REVERSE # Identifies reversed EPI readouts.
using LinearAlgebra: norm # Computes optimizer step norms.
@eval using $(Sys.isapple() ? :Metal : :CUDA) # Selects the available GPU backend.

archive_directory = isempty(ARGS) ? joinpath(homedir(), "Desktop/Archive (1)") : first(ARGS) # Allows a custom data root.
fully_sampled_mrd_file = joinpath( # Builds the ESPIRiT reference path.
    archive_directory, # Anchors the path at the data root.
    "mrd_hdf5/meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd", # Selects the full scan.
) # Completes the reference path.
accelerated_mrd_file = joinpath( # Builds the measured-data path.
    archive_directory, # Anchors the path at the data root.
    "mrd_hdf5/meas_MID01109_FID34203_hard_epi_2x_20interleaves_5avg_fatsat.mrd", # Selects the R=2 scan.
) # Completes the measured-data path.
sequence_file = joinpath(archive_directory, "seq/hard_epi_2x_20interleaves_5avg_fatsat.seq") # Uses the matching sequence.
iteration_directory = joinpath(@__DIR__, "Diff8x8") # Keeps outputs beside the script.
mkpath(iteration_directory) # Ensures the output folder exists.

recon_size = (128, 128) # Matches the acquired Cartesian matrix.
voxel_grid = (8, 8, 1) # Defines the unknown density grid.
subspin_grid = (5, 2, 1) # Uses ten simulation spins per voxel.
simulation_spins_per_voxel = prod(subspin_grid) # Preserves density when subdividing voxels.
navigator_count = 3 # Matches the three leading EPI navigators.
brain_T1 = 1.0 # Adds fixed longitudinal relaxation in seconds.
brain_T2 = 0.1 # Adds fixed transverse relaxation in seconds.
brain_T2s = 0.05 # Adds fixed effective transverse relaxation in seconds.
finite_difference_step = cbrt(eps(Float32)) # Matches the DiffCross gradient step.
maximum_iterations = 20 # Matches the DiffCross iteration budget.
λ₀ = 1e-6 # Matches the adaptive-gradient initial step.

seq = read_seq(sequence_file) # Loads the exact accelerated pulse sequence.
shots = Int(seq.DEF["EpiShots"]) # Reads the number of EPI interleaves.
acquired_shots = parse.(Int, split(seq.DEF["EpiAcquiredShots"], ',')) .- 1 # Converts retained shots to zero-based indices.
acquired_lines = [line for shot in acquired_shots for line in shot:shots:(recon_size[2] - 1)] # Restores acquisition-order ky lines.

raw_measured = RawAcquisitionData(ISMRMRDFile(accelerated_mrd_file)) # Loads the measured R=2 profiles.
navigator_profile_indices = 1:navigator_count # Separates navigators once.
image_profile_indices = (navigator_count + 1):(navigator_count + length(acquired_lines)) # Separates the first image average once.
navigator_profiles = raw_measured.profiles[navigator_profile_indices] # Keeps only navigator profiles.
image_profiles = raw_measured.profiles[image_profile_indices] # Keeps only image profiles.
measured_navigators = ComplexF32.(reduce(vcat, profile.data for profile in navigator_profiles)) # Stacks navigator samples by acquisition order.
measured_image = ComplexF32.(reduce(vcat, profile.data for profile in image_profiles)) # Stacks image samples by acquisition order.

function correct_navigator_phase(navigators, image, profiles) # Removes odd/even EPI phase mismatch.
    samples = size(navigators, 1) ÷ 3 # Recovers samples in each navigator.
    navigation = reshape(navigators, samples, 3, :) # Separates the three navigator readouts.
    centered(transform, data) = KomaMRI.fftshift(transform(KomaMRI.ifftshift(data, 1), 1), 1) # Centers readout transforms.
    forward = centered(KomaMRI.ifft, (navigation[:, 1, :] .+ navigation[:, 3, :]) ./ 2) # Averages the two forward navigators.
    reverse_aligned = centered(KomaMRI.ifft, reverse(navigation[:, 2, :]; dims=1)) # Aligns the reverse navigator direction.
    correction = exp.(-im .* angle.(vec(sum(conj.(forward) .* reverse_aligned; dims=2)))) # Estimates their shared spatial phase error.

    readouts = reshape(copy(image), samples, length(profiles), :) # Exposes sample, profile, and coil axes.
    reverse_profiles = map(profile -> !iszero(profile.head.flags & ISMRMRD_ACQ_IS_REVERSE), profiles) # Finds reversed image readouts.
    aligned = reverse(readouts[:, reverse_profiles, :]; dims=1) # Aligns those readouts before correction.
    corrected = centered(KomaMRI.ifft, aligned) .* reshape(correction, :, 1, 1) # Applies correction in readout-image space.
    readouts[:, reverse_profiles, :] .= reverse(centered(KomaMRI.fft, corrected); dims=1) # Restores measured k-space order.
    reshape(readouts, size(image)) # Returns the original samples-by-coils layout.
end # Completes navigator correction.

b = correct_navigator_phase(measured_navigators, measured_image, image_profiles) # Sets the corrected reconstruction target.

navigator_sample_indices = axes(measured_navigators, 1) # Marks simulated navigator rows once.
image_sample_indices = (length(navigator_sample_indices) + 1):(length(navigator_sample_indices) + size(b, 1)) # Marks simulated image rows once.
adc_blocks = findall(block -> is_ADC_on(seq[block]), 1:length(seq)) # Locates sequence readout blocks.
seq = seq[1:adc_blocks[navigator_count + length(acquired_lines)]] # Removes unused later averages.

raw_reference = RawAcquisitionData(ISMRMRDFile(fully_sampled_mrd_file)) # Loads the fully sampled reference.
raw_reference.profiles = raw_reference.profiles[(navigator_count + 1):(navigator_count + recon_size[2])] # Keeps its first image average.
acq_reference = AcquisitionData(raw_reference) # Places reference profiles on a Cartesian grid.
acq_reference.traj[1].circular = false # Prevents an incorrect circular shutter.
sensitivity_maps = espirit( # Estimates receive sensitivities from measured data.
    acq_reference, # Supplies the fully sampled reference.
    (6, 6), # Uses the documented ESPIRiT kernel.
    30, # Uses the documented calibration width.
    recon_size; # Returns maps on the acquisition matrix.
    eigThresh_1=0.02, # Keeps stable calibration singular vectors.
    eigThresh_2=0.0, # Retains the complete spatial map support.
) # Completes ESPIRiT estimation.

map_fov = Float32.(raw_reference.params["reconFOV"]) .* 1f-3 # Converts the measured FOV from millimetres to metres.
map_x = collect(LinRange(-map_fov[1] / 2, map_fov[1] / 2, recon_size[1])) # Places map samples across x.
map_y = collect(LinRange(-map_fov[2] / 2, map_fov[2] / 2, recon_size[2])) # Places map samples across y.
map_z = Float32[-map_fov[3] / 2, 0, map_fov[3] / 2] # Extends 2D maps through the slice.
receiver = ArbitraryCoilSens( # Converts ESPIRiT maps into a KomaMRI receiver.
    map_x, # Supplies map x coordinates.
    map_y, # Supplies map y coordinates.
    map_z, # Supplies map z coordinates.
    repeat(sensitivity_maps, 1, 1, length(map_z), 1), # Repeats maps without changing coils.
) # Completes the receive model.
scanner = Scanner(; receiver) # Encodes measured coil information in simulation.
fov = Float64.(map_fov) # Matches Phantom's Float64 coordinate type.

centered_axis(width, count) = range(-width / 2 + width / (2count); step=width / count, length=count) # Places samples at cell centres.
voxel_count = prod(voxel_grid) # Counts the unknown densities.
voxel_axis_x, voxel_axis_y = centered_axis.(fov[1:2], voxel_grid[1:2]) # Locates the 8 × 8 density values.
simulation_grid = voxel_grid .* subspin_grid # Expands each voxel into ten spins.
simulation_axes = centered_axis.(fov, simulation_grid) # Locates every simulation spin.
spin_points = vec(collect(Iterators.product(simulation_axes...))) # Flattens the 3D spin grid.
spin_x = first.(spin_points) # Supplies Phantom x coordinates.
spin_y = getindex.(spin_points, 2) # Supplies Phantom y coordinates.
spin_z = last.(spin_points) # Supplies Phantom z coordinates.
total_spin_count = length(spin_points) # Sizes all spin-wise tissue arrays.

function simulation_object(x, params) # Converts guessed voxel densities into a Phantom.
    interpolation = linear_interpolation( # Builds the voxel-to-spin density map.
        (params.voxel_axis_x, params.voxel_axis_y), reshape(x, params.voxel_grid[1:2]); # Supplies density locations and values.
        extrapolation_bc=Flat(), # Avoids edge extrapolation artifacts.
    ) # Completes the density interpolator.
    Phantom(; # Builds the object simulated by KomaMRI.
        name="8 x 8 in-vivo brain reconstruction", # Labels simulation output.
        x=params.spin_x, # Sets spin x coordinates.
        y=params.spin_y, # Sets spin y coordinates.
        z=params.spin_z, # Sets spin z coordinates.
        ρ=interpolation.(params.spin_x, params.spin_y) ./ params.simulation_spins_per_voxel, # Distributes voxel density across subspins.
        T1=fill(params.brain_T1, params.total_spin_count), # Applies fixed T1 to every spin.
        T2=fill(params.brain_T2, params.total_spin_count), # Applies fixed T2 to every spin.
        T2s=fill(params.brain_T2s, params.total_spin_count), # Applies fixed T2* to every spin.
    ) # Returns the configured Phantom.
end # Completes object construction.

sim_params = Dict{String,Any}("gpu" => true, "precision" => "f32", "sim_method" => Bloch(), "return_type" => "mat") # Requests GPU Bloch signals as a matrix.
physio = CardiacSignal(; heart_rate=1) # Satisfies the triggered sequence timing.
params = (; # Collects constants excluded from optimization.
    seq, scanner, sim_params, physio, b, image_sample_indices, voxel_grid, voxel_axis_x, voxel_axis_y, # Stores acquisition and voxel geometry.
    spin_x, spin_y, spin_z, total_spin_count, simulation_spins_per_voxel, brain_T1, brain_T2, # Stores spin geometry and fixed tissue values.
    brain_T2s, finite_difference_step, maximum_iterations, λ₀, iteration_directory, # Stores the remaining fixed controls.
) # Completes the immutable parameter bundle.

function simulate_image(x, params) # Simulates only the measured image portion.
    signal = simulate( # Runs the physical KomaMRI forward model.
        simulation_object(x, params), params.seq, params.scanner; # Supplies object, sequence, and measured coils.
        sim_params=params.sim_params, physio=params.physio, verbose=false, # Uses GPU settings without repeated logs.
    ) # Returns navigator and image samples.
    @view signal[params.image_sample_indices, :] # Excludes simulated navigators from the loss.
end # Completes the real-density simulation.

J = FiniteDiff.finite_difference_jacobian( # Calibrates one shared object phase before reconstruction.
    x -> vec(simulate_image(x, params)), zeros(voxel_count), Val(:forward), ComplexF32; absstep=1.0, # Uses unit columns because density is linear.
) # Completes the one-time calibration Jacobian.
complex_density = (J' * J) \ (J' * vec(params.b)) # Finds the complex density used only for phase and scale.
density_scale = maximum(abs, complex_density) # Uses one global scale for ESPIRiT's arbitrary amplitude.
spatial_phase = complex_density ./ max.(abs.(complex_density), eps(Float32)) # Keeps one phase per voxel, shared by all coils.
params = (; params..., b=params.b ./ density_scale, spatial_phase) # Fixes phase and scale before optimization.

A(x, params) = # Predicts data for a real nonnegative density magnitude.
    simulate_image(real.(params.spatial_phase .* x), params) .+ # Simulates the in-phase component.
    im .* simulate_image(imag.(params.spatial_phase .* x), params) # Adds the quadrature component without coil gains.
f(x, params)::Float32 = sum(abs2, A(x, params) - params.b) # Uses the DiffCross squared residual.

save_density_image(x, name, title, params) = savefig( # Saves each reconstruction state.
    plot_image(reshape(x, params.voxel_grid[1:2]); title), joinpath(params.iteration_directory, name), # Plots the 8 × 8 density in its output folder.
) # Completes image saving.

save_density_image(abs.(complex_density) ./ density_scale, "phase_calibration.png", "Shared spatial-phase calibration", params) # Records the fixed phase-calibration magnitude.

function reconstruct(x, params) # Runs the DiffCross adaptive finite-difference optimizer.
    objective(x) = f(x, params) # Holds every non-density parameter fixed.
    save_density_image(x, "iteration_00.png", "Iteration 0", params) # Records the zero initialization.
    ∇fₖ = FiniteDiff.finite_difference_gradient(objective, x; relstep=params.finite_difference_step) # Computes the initial gradient.
    xₖ₋₁ = copy(x) # Stores the previous density vector.
    ∇fₖ₋₁ = ∇fₖ # Stores the previous gradient.
    λₖ = params.λ₀ # Starts from the reference step size.
    θₖ = Inf # Leaves the first adaptive bound unrestricted.

    for iteration in 1:params.maximum_iterations # Applies at most twenty updates.
        if iteration > 1 # Waits for two iterate-gradient pairs.
            ∇fₖ = FiniteDiff.finite_difference_gradient(objective, x; relstep=params.finite_difference_step) # Recomputes the current gradient.
            Δxₖ = norm(x - xₖ₋₁) # Measures the iterate change.
            Δ∇fₖ = norm(∇fₖ - ∇fₖ₋₁) # Measures the local gradient change.
            λₖ₋₁, θₖ₋₁ = λₖ, θₖ # Preserves the previous adaptive state.
            λₖ = min(sqrt(1 + θₖ₋₁) * λₖ₋₁, iszero(Δ∇fₖ) ? λₖ₋₁ : Δxₖ / (2Δ∇fₖ)) # Limits the step using local curvature.
            θₖ = λₖ / λₖ₋₁ # Updates the next growth bound.
        end # Completes step adaptation.
        ∇fₖ_norm = norm(∇fₖ) # Measures the available descent direction.
        iszero(∇fₖ_norm) && break # Stops when the gradient vanishes.
        xₖ₋₁ .= x # Saves the current density before updating it.
        ∇fₖ₋₁ = ∇fₖ # Saves the gradient paired with that density.
        x .-= λₖ .* ∇fₖ # Applies the DiffCross gradient step.

        save_density_image(x, "iteration_$(lpad(iteration, 2, '0')).png", "Iteration $iteration", params) # Records this iterate.
        println("Iteration $iteration: loss = $(objective(x)), gradient = $∇fₖ_norm, λ = $λₖ") # Reports convergence values.
    end # Completes the iteration loop.

    save_density_image(x, "reconstructed_density.png", "Reconstructed brain density", params) # Saves the final density.
    println("Final loss: ", objective(x)) # Reports the final data mismatch.
    println("Density extrema: ", extrema(x)) # Reports the recovered density range.
    println("Saved images to: ", params.iteration_directory) # Reports the output location.
    x # Returns the reconstructed density.
end # Completes reconstruction.

reconstruct(zeros(voxel_count), params) # Starts from no assumed brain density.
