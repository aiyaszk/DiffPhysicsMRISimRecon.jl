# Finite-difference reconstruction of an 128 x 128 density from an accelerated in-vivo brain acquisition

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using FiniteDiff, KomaMRI, MRICoilSensitivities
using Interpolations: Flat, linear_interpolation
using KomaMRIBase: ArbitraryCoilSens, Phantom
using KomaMRICore: ISMRMRD_ACQ_IS_REVERSE
using LinearAlgebra: norm
@eval using $(Sys.isapple() ? :Metal : :CUDA)

archive_directory = isempty(ARGS) ? joinpath(homedir(), "Desktop/Archive (1)") : first(ARGS)
fully_sampled_mrd_file = joinpath(
    archive_directory,
    "mrd_hdf5/meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd",
)
accelerated_mrd_file = joinpath(
    archive_directory,
    "mrd_hdf5/meas_MID01109_FID34203_hard_epi_2x_20interleaves_5avg_fatsat.mrd",
)
sequence_file = joinpath(archive_directory, "seq/hard_epi_2x_20interleaves_5avg_fatsat.seq")
iteration_directory = joinpath(@__DIR__, "Diff128x128")
mkpath(iteration_directory)

recon_size = (128, 128)
voxel_grid = (128, 128, 1)
subspin_grid = (5, 2, 1)
simulation_spins_per_voxel = prod(subspin_grid)
navigator_count = 3
brain_T1 = 1.0
brain_T2 = 0.1
brain_T2s = 0.05
finite_difference_step = cbrt(eps(Float32))
maximum_iterations = 20
λ₀ = 1e-6 # Uses the default from the adaptive-gradient reference implementation.

seq = read_seq(sequence_file)
shots = Int(seq.DEF["EpiShots"])
acquired_shots = parse.(Int, split(seq.DEF["EpiAcquiredShots"], ',')) .- 1
acquired_lines = [line for shot in acquired_shots for line in shot:shots:(recon_size[2] - 1)]

raw_measured = RawAcquisitionData(ISMRMRDFile(accelerated_mrd_file))
navigator_profile_indices = 1:navigator_count
image_profile_indices = (navigator_count + 1):(navigator_count + length(acquired_lines))
navigator_profiles = raw_measured.profiles[navigator_profile_indices]
image_profiles = raw_measured.profiles[image_profile_indices]
measured_navigators = ComplexF32.(reduce(vcat, profile.data for profile in navigator_profiles))
measured_image = ComplexF32.(reduce(vcat, profile.data for profile in image_profiles))

function correct_navigator_phase(navigators, image, profiles)
    samples = size(navigators, 1) ÷ 3
    navigation = reshape(navigators, samples, 3, :)
    centered(transform, data) = KomaMRI.fftshift(transform(KomaMRI.ifftshift(data, 1), 1), 1)
    forward = centered(KomaMRI.ifft, (navigation[:, 1, :] .+ navigation[:, 3, :]) ./ 2)
    reverse_aligned = centered(KomaMRI.ifft, reverse(navigation[:, 2, :]; dims=1))
    correction = exp.(-im .* angle.(vec(sum(conj.(forward) .* reverse_aligned; dims=2))))

    readouts = reshape(copy(image), samples, length(profiles), :)
    reverse_profiles = map(profile -> !iszero(profile.head.flags & ISMRMRD_ACQ_IS_REVERSE), profiles)
    aligned = reverse(readouts[:, reverse_profiles, :]; dims=1)
    corrected = centered(KomaMRI.ifft, aligned) .* reshape(correction, :, 1, 1)
    readouts[:, reverse_profiles, :] .= reverse(centered(KomaMRI.fft, corrected); dims=1)
    reshape(readouts, size(image))
end

b = correct_navigator_phase(measured_navigators, measured_image, image_profiles)

navigator_sample_indices = axes(measured_navigators, 1)
image_sample_indices = (length(navigator_sample_indices) + 1):(length(navigator_sample_indices) + size(b, 1))
adc_blocks = findall(block -> is_ADC_on(seq[block]), 1:length(seq))
seq = seq[1:adc_blocks[navigator_count + length(acquired_lines)]]

raw_reference = RawAcquisitionData(ISMRMRDFile(fully_sampled_mrd_file))
raw_reference.profiles = raw_reference.profiles[(navigator_count + 1):(navigator_count + recon_size[2])]
acq_reference = AcquisitionData(raw_reference)
acq_reference.traj[1].circular = false
sensitivity_maps = espirit(
    acq_reference,
    (6, 6),
    30,
    recon_size;
    eigThresh_1=0.02,
    eigThresh_2=0.0,
)

map_fov = Float32.(raw_reference.params["reconFOV"]) .* 1f-3
map_x = collect(LinRange(-map_fov[1] / 2, map_fov[1] / 2, recon_size[1]))
map_y = collect(LinRange(-map_fov[2] / 2, map_fov[2] / 2, recon_size[2]))
map_z = Float32[-map_fov[3] / 2, 0, map_fov[3] / 2]
receiver = ArbitraryCoilSens(
    map_x,
    map_y,
    map_z,
    repeat(sensitivity_maps, 1, 1, length(map_z), 1),
)
scanner = Scanner(; receiver)
fov = Float64.(map_fov)

centered_axis(width, count) = range(-width / 2 + width / (2count); step=width / count, length=count)
voxel_count = prod(voxel_grid)
voxel_axis_x, voxel_axis_y = centered_axis.(fov[1:2], voxel_grid[1:2])
simulation_grid = voxel_grid .* subspin_grid
simulation_axes = centered_axis.(fov, simulation_grid)
spin_points = vec(collect(Iterators.product(simulation_axes...)))
spin_x = first.(spin_points)
spin_y = getindex.(spin_points, 2)
spin_z = last.(spin_points)
total_spin_count = length(spin_points)

function simulation_object(x, params)
    interpolation = linear_interpolation(
        (params.voxel_axis_x, params.voxel_axis_y), reshape(x, params.voxel_grid[1:2]);
        extrapolation_bc=Flat(),
    )
    Phantom(;
        name="128 x 128 in-vivo brain reconstruction",
        x=params.spin_x,
        y=params.spin_y,
        z=params.spin_z,
        ρ=interpolation.(params.spin_x, params.spin_y) ./ params.simulation_spins_per_voxel,
        T1=fill(params.brain_T1, params.total_spin_count),
        T2=fill(params.brain_T2, params.total_spin_count),
        T2s=fill(params.brain_T2s, params.total_spin_count),
    )
end

sim_params = Dict{String,Any}("gpu" => true, "precision" => "f32", "sim_method" => Bloch(), "return_type" => "mat")
physio = CardiacSignal(; heart_rate=1)
params = (;
    seq, scanner, sim_params, physio, b, image_sample_indices, voxel_grid, voxel_axis_x, voxel_axis_y,
    spin_x, spin_y, spin_z, total_spin_count, simulation_spins_per_voxel, brain_T1, brain_T2,
    brain_T2s, finite_difference_step, maximum_iterations, λ₀, iteration_directory,
)

function simulate_image(x, params)
    signal = simulate(
        simulation_object(x, params), params.seq, params.scanner;
        sim_params=params.sim_params, physio=params.physio, verbose=false,
    )
    @view signal[params.image_sample_indices, :]
end

J = FiniteDiff.finite_difference_jacobian(
    x -> vec(simulate_image(x, params)), zeros(voxel_count), Val(:forward), ComplexF32; absstep=1.0,
)
complex_density = (J' * J) \ (J' * vec(params.b))
density_scale = maximum(abs, complex_density)
spatial_phase = complex_density ./ max.(abs.(complex_density), eps(Float32))
params = (; params..., b=params.b ./ density_scale, spatial_phase)

A(x, params) =
    simulate_image(real.(params.spatial_phase .* x), params) .+
    im .* simulate_image(imag.(params.spatial_phase .* x), params)
f(x, params)::Float32 = sum(abs2, A(x, params) - params.b)

save_density_image(x, name, title, params) = savefig(
    plot_image(reshape(x, params.voxel_grid[1:2]); title), joinpath(params.iteration_directory, name),
)

save_density_image(abs.(complex_density) ./ density_scale, "phase_calibration.png", "Shared spatial-phase calibration", params)

function reconstruct(x, params)
    objective(x) = f(x, params)
    save_density_image(x, "iteration_00.png", "Iteration 0", params)
    ∇fₖ = FiniteDiff.finite_difference_gradient(objective, x; relstep=params.finite_difference_step)
    xₖ₋₁ = copy(x)
    ∇fₖ₋₁ = ∇fₖ
    λₖ = params.λ₀
    θₖ = Inf

    for iteration in 1:params.maximum_iterations
        if iteration > 1
            ∇fₖ = FiniteDiff.finite_difference_gradient(objective, x; relstep=params.finite_difference_step)
            Δxₖ = norm(x - xₖ₋₁)
            Δ∇fₖ = norm(∇fₖ - ∇fₖ₋₁)
            λₖ₋₁, θₖ₋₁ = λₖ, θₖ
            λₖ = min(sqrt(1 + θₖ₋₁) * λₖ₋₁, iszero(Δ∇fₖ) ? λₖ₋₁ : Δxₖ / (2Δ∇fₖ))
            θₖ = λₖ / λₖ₋₁
        end
        ∇fₖ_norm = norm(∇fₖ)
        iszero(∇fₖ_norm) && break
        xₖ₋₁ .= x
        ∇fₖ₋₁ = ∇fₖ
        x .-= λₖ .* ∇fₖ

        save_density_image(x, "iteration_$(lpad(iteration, 2, '0')).png", "Iteration $iteration", params)
        println("Iteration $iteration: loss = $(objective(x)), gradient = $∇fₖ_norm, λ = $λₖ")
    end

    save_density_image(x, "reconstructed_density.png", "Reconstructed brain density", params)
    println("Final loss: ", objective(x))
    println("Density extrema: ", extrema(x))
    println("Saved images to: ", params.iteration_directory)
    x
end

reconstruct(zeros(voxel_count), params)
