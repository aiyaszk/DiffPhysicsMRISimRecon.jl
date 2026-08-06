using FiniteDiff
using Interpolations: Flat, linear_interpolation
using KomaMRI
using KomaMRIBase: ArbitraryCoilSens, Phantom
using LinearAlgebra: norm
using MRICoilSensitivities
@eval using $(Sys.isapple() ? :Metal : :CUDA)

centered_axis(width, count) =
    range(-width / 2 + width / (2count); step=width / count, length=count)

function simulation_object(x, params)
    interpolation = linear_interpolation(
        (params.voxel_axis_x, params.voxel_axis_y),
        reshape(x, params.voxel_grid[1:2]);
        extrapolation_bc=Flat(),
    )
    return Phantom(;
        name=params.object_name,
        x=params.spin_x,
        y=params.spin_y,
        z=params.spin_z,
        ρ=interpolation.(params.spin_x, params.spin_y) ./ params.simulation_spins_per_voxel,
        T1=fill(params.brain_T1, params.total_spin_count),
        T2=fill(params.brain_T2, params.total_spin_count),
        T2s=fill(params.brain_T2s, params.total_spin_count),
    )
end

function simulate_image(x, params)
    signal = simulate(
        simulation_object(x, params),
        params.seq,
        params.scanner;
        sim_params=params.sim_params,
        physio=params.physio,
        verbose=false,
    )
    return @view signal[params.image_sample_indices, :]
end

A(x, params) = simulate_image(x, params)
f(x, params)::Float32 = sum(abs2, A(x, params) - params.b)

save_density_image(x, name, title, params) = savefig(
    plot_image(reshape(x, params.voxel_grid[1:2]); title),
    joinpath(params.iteration_directory, name),
)

function reconstruct(x, params)
    objective(x) = f(x, params)
    save_density_image(x, "iteration_00.png", "Iteration 0", params)
    gradient = FiniteDiff.finite_difference_gradient(
        objective,
        x;
        relstep=params.finite_difference_step,
    )
    previous_x = copy(x)
    previous_gradient = gradient
    step = params.initial_step
    step_ratio = Inf

    for iteration in 1:params.maximum_iterations
        if iteration > 1
            gradient = FiniteDiff.finite_difference_gradient(
                objective,
                x;
                relstep=params.finite_difference_step,
            )
            density_change = norm(x - previous_x)
            gradient_change = norm(gradient - previous_gradient)
            previous_step, previous_ratio = step, step_ratio
            step = min(
                sqrt(1 + previous_ratio) * previous_step,
                iszero(gradient_change) ? previous_step : density_change / (2gradient_change),
            )
            step_ratio = step / previous_step
        end

        gradient_norm = norm(gradient)
        iszero(gradient_norm) && break
        previous_x .= x
        previous_gradient = gradient
        x .-= step .* gradient

        save_density_image(
            x,
            "iteration_$(lpad(iteration, 2, '0')).png",
            "Iteration $iteration",
            params,
        )
        println(
            "Iteration $iteration: loss = $(objective(x)), " *
            "gradient = $gradient_norm, step = $step",
        )
    end

    save_density_image(x, "reconstructed_density.png", "Reconstructed brain density", params)
    println("Final loss: ", objective(x))
    println("Density extrema: ", extrema(x))
    println("Saved images to: ", params.iteration_directory)
    return x
end

function run_finite_difference_reconstruction(voxel_grid)
    archive_directory =
        isempty(ARGS) ? joinpath(homedir(), "Desktop/Archive (1)") : first(ARGS)
    fully_sampled_mrd_file = joinpath(
        archive_directory,
        "mrd_hdf5/meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd",
    )
    accelerated_mrd_file = joinpath(
        archive_directory,
        "mrd_hdf5/meas_MID01109_FID34203_hard_epi_2x_20interleaves_5avg_fatsat.mrd",
    )
    sequence_file =
        joinpath(archive_directory, "seq/hard_epi_2x_20interleaves_5avg_fatsat.seq")
    resolution = first(voxel_grid)
    iteration_directory = joinpath(@__DIR__, "Diff$(resolution)x$(resolution)")
    mkpath(iteration_directory)

    recon_size = (128, 128)
    subspin_grid = (5, 2, 1)
    simulation_spins_per_voxel = prod(subspin_grid)
    navigator_count = 3
    brain_T1 = 1.0
    brain_T2 = 0.1
    brain_T2s = 0.05
    finite_difference_step = cbrt(eps(Float32))
    maximum_iterations = 20
    initial_step = 1e-6

    seq = read_seq(sequence_file)
    shots = Int(seq.DEF["EpiShots"])
    acquired_shots = parse.(Int, split(seq.DEF["EpiAcquiredShots"], ',')) .- 1
    acquired_lines = [
        line for shot in acquired_shots for line in shot:shots:(recon_size[2] - 1)
    ]

    raw_measured = RawAcquisitionData(ISMRMRDFile(accelerated_mrd_file))
    image_profile_indices =
        (navigator_count + 1):(navigator_count + length(acquired_lines))
    image_profiles = raw_measured.profiles[image_profile_indices]
    b = ComplexF32.(reduce(vcat, profile.data for profile in image_profiles))

    samples_per_profile = size(first(image_profiles).data, 1)
    image_sample_offset = navigator_count * samples_per_profile
    image_sample_indices =
        (image_sample_offset + 1):(image_sample_offset + size(b, 1))
    adc_blocks = findall(block -> is_ADC_on(seq[block]), 1:length(seq))
    seq = seq[1:adc_blocks[navigator_count + length(acquired_lines)]]

    raw_reference = RawAcquisitionData(ISMRMRDFile(fully_sampled_mrd_file))
    raw_reference.profiles =
        raw_reference.profiles[(navigator_count + 1):(navigator_count + recon_size[2])]
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
    coil_intensity = sqrt.(sum(abs2, sensitivity_maps; dims=4))
    sensitivity_maps ./= max.(coil_intensity, eps(Float32))

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

    voxel_count = prod(voxel_grid)
    voxel_axis_x, voxel_axis_y = centered_axis.(fov[1:2], voxel_grid[1:2])
    simulation_grid = voxel_grid .* subspin_grid
    simulation_axes = centered_axis.(fov, simulation_grid)
    spin_points = vec(collect(Iterators.product(simulation_axes...)))
    spin_x = first.(spin_points)
    spin_y = getindex.(spin_points, 2)
    spin_z = last.(spin_points)
    total_spin_count = length(spin_points)
    object_name = "$(resolution) x $(resolution) in-vivo brain reconstruction"
    sim_params = Dict{String,Any}(
        "gpu" => true,
        "precision" => "f32",
        "sim_method" => Bloch(),
        "return_type" => "mat",
    )
    physio = CardiacSignal(; heart_rate=1)
    params = (;
        seq,
        scanner,
        sim_params,
        physio,
        b,
        image_sample_indices,
        voxel_grid,
        voxel_axis_x,
        voxel_axis_y,
        spin_x,
        spin_y,
        spin_z,
        total_spin_count,
        simulation_spins_per_voxel,
        brain_T1,
        brain_T2,
        brain_T2s,
        finite_difference_step,
        maximum_iterations,
        initial_step,
        iteration_directory,
        object_name,
    )
    return reconstruct(zeros(voxel_count), params)
end
