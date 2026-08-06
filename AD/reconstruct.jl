using Enzyme
using KomaMRI
using KomaMRIBase: ArbitraryCoilSens, Phantom, get_sens
using KomaMRICore: ISMRMRD_ACQ_IS_REVERSE
using LinearAlgebra: dot, norm
using MRICoilSensitivities
using Reactant
@eval using $(Sys.isapple() ? :Metal : :CUDA)

isdefined(Main, :ADSequence) || include("ad_multicoil.jl")

Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", Sys.isapple() ? "cpu" : "cuda"))
Reactant.allowscalar(false)

centered_axis(width, count) =
    range(-width / 2 + width / (2count); step=width / count, length=count)

function correct_navigator_phase(navigators, image, profiles)
    samples = size(navigators, 1) ÷ 3
    navigation = reshape(navigators, samples, 3, :)
    centered(transform, data) =
        KomaMRI.fftshift(transform(KomaMRI.ifftshift(data, 1), 1), 1)
    forward = centered(KomaMRI.ifft, (navigation[:, 1, :] .+ navigation[:, 3, :]) ./ 2)
    reverse_aligned = centered(KomaMRI.ifft, reverse(navigation[:, 2, :]; dims=1))
    correction = exp.(-im .* angle.(vec(sum(conj.(forward) .* reverse_aligned; dims=2))))

    readouts = reshape(copy(image), samples, length(profiles), :)
    reverse_profiles =
        map(profile -> !iszero(profile.head.flags & ISMRMRD_ACQ_IS_REVERSE), profiles)
    aligned = reverse(readouts[:, reverse_profiles, :]; dims=1)
    corrected = centered(KomaMRI.ifft, aligned) .* reshape(correction, :, 1, 1)
    readouts[:, reverse_profiles, :] .= reverse(centered(KomaMRI.fft, corrected); dims=1)
    return reshape(readouts, size(image))
end

function spin_object(density, params)
    return Phantom(;
        name=params.object_name,
        x=params.spin_x,
        y=params.spin_y,
        z=params.spin_z,
        ρ=density,
        T1=fill(params.brain_T1, params.total_spin_count),
        T2=fill(params.brain_T2, params.total_spin_count),
        T2s=fill(params.brain_T2s, params.total_spin_count),
    )
end

function koma_signal(x, params)
    signal = simulate(
        spin_object(
            interpolate_voxels(x, params.stencil) ./ params.simulation_spins_per_voxel,
            params,
        ),
        params.seq,
        params.scanner;
        sim_params=params.sim_params,
        physio=params.physio,
        verbose=false,
    )
    return @view signal[params.image_sample_indices, :]
end

function simulate_reference_response(params)
    reference_params = merge(
        params,
        (;
            spin_x=zeros(Float64, 1),
            spin_y=zeros(Float64, 1),
            spin_z=zeros(Float64, 1),
            total_spin_count=1,
        ),
    )
    signal = simulate(
        spin_object(ones(Float32, 1), reference_params),
        params.seq,
        Scanner();
        sim_params=params.sim_params,
        physio=params.physio,
        verbose=false,
    )
    return vec(Array(@view signal[params.image_sample_indices, :]))
end

function fft_encoding(params, receiver_values)
    settings = KomaMRICore.default_sim_params(copy(params.sim_params))
    sampling_rule = KomaMRICore.simulation_sampling_rule(settings["sim_method"], settings)
    seq = KomaMRIBase.resolve_triggers(params.seq, params.physio)
    _, trajectory = get_kspace(seq; sampling_rule)
    trajectory = trajectory[params.image_sample_indices, :]
    modes_float = trajectory .* reshape(params.fov, 1, :)
    modes = round.(Int, modes_float)
    mode_error = maximum(abs, modes_float - modes)
    println("Cartesian trajectory mode error: ", mode_error)
    mode_error < 5e-3 || error("Sequence trajectory is not Cartesian")

    Nx, Ny, Nz = params.simulation_grid
    sample_indices = @. 1 + mod(modes[:, 1], Nx) +
                        mod(modes[:, 2], Ny) * Nx +
                        mod(modes[:, 3], Nz) * Nx * Ny
    offsets = collect(1 .- inv.(Float64.((Nx, Ny, Nz))))
    center_phase = exp.(ComplexF32(0, π) .* vec(sum(modes .* reshape(offsets, 1, :); dims=2)))
    sample_weight = ComplexF32.(simulate_reference_response(params) .* center_phase)
    return (;
        sample_indices,
        sample_weight,
        receiver_values,
        stencil=params.stencil,
        simulation_grid=params.simulation_grid,
        simulation_spins_per_voxel=params.simulation_spins_per_voxel,
        voxel_count=params.voxel_count,
    )
end

function fft_signal(x, model)
    spin_density = interpolate_voxels(x, model.stencil) ./
                   model.simulation_spins_per_voxel
    spin_coils = reshape(spin_density, :, 1) .* model.receiver_values
    grid_coils = reshape(spin_coils, model.simulation_grid..., size(spin_coils, 2))
    spectrum = KomaMRI.fft(grid_coils, (1, 2, 3))
    sampled = reshape(spectrum, :, size(spin_coils, 2))[model.sample_indices, :]
    return reshape(model.sample_weight, :, 1) .* sampled
end

function interpolation_adjoint(spin_values, stencil, voxel_count)
    voxel_values = zeros(eltype(spin_values), voxel_count)
    for spin in eachindex(spin_values)
        wx = stencil.wx[spin]
        wy = stencil.wy[spin]
        voxel_values[stencil.i00[spin]] += (1 - wx) * (1 - wy) * spin_values[spin]
        voxel_values[stencil.i10[spin]] += wx * (1 - wy) * spin_values[spin]
        voxel_values[stencil.i01[spin]] += (1 - wx) * wy * spin_values[spin]
        voxel_values[stencil.i11[spin]] += wx * wy * spin_values[spin]
    end
    return voxel_values
end

function fft_adjoint(signal, model)
    spectrum = zeros(eltype(signal), prod(model.simulation_grid), size(signal, 2))
    weighted_signal = conj.(model.sample_weight) .* signal
    for sample in axes(signal, 1)
        spectrum[model.sample_indices[sample], :] .+= @view weighted_signal[sample, :]
    end
    grid_spectrum = reshape(spectrum, model.simulation_grid..., size(signal, 2))
    spin_coils = prod(model.simulation_grid) .* KomaMRI.ifft(grid_spectrum, (1, 2, 3))
    receiver_grid = reshape(
        model.receiver_values,
        model.simulation_grid...,
        size(model.receiver_values, 2),
    )
    spin_adjoint = vec(sum(conj.(receiver_grid) .* spin_coils; dims=4))
    return interpolation_adjoint(spin_adjoint, model.stencil, model.voxel_count) ./
           model.simulation_spins_per_voxel
end

A(x, model) = fft_signal(model.spatial_phase .* x, model)

f(x, model) = sum(abs2, A(x, model) - model.b)

function f_gradient(x, model)
    result = Enzyme.gradient(
        Enzyme.ReverseWithPrimal,
        f,
        x,
        Enzyme.Const(model),
    )
    return result.val, result.derivs[1]
end

save_density_image(x, name, title, params) = savefig(
    plot_image(reshape(x, params.voxel_grid[1:2]); title),
    joinpath(params.iteration_directory, name),
)

function reconstruct(x, controls, model, gradient_function, loss_function, initial_state)
    save_density_image(Array(x), "iteration_00.png", "Iteration 0", controls)
    loss, gradient = initial_state
    println("Initial loss: ", Reactant.to_number(loss))
    λₖ = controls.λ₀
    θₖ = Float32(Inf)

    for iteration in 1:controls.maximum_iterations
        gradient_norm = norm(Array(gradient))
        iszero(gradient_norm) && break
        previous_x = x
        previous_gradient = gradient
        x = x .- λₖ .* gradient

        if iteration < controls.maximum_iterations
            loss, gradient = gradient_function(x, model)
        else
            loss = loss_function(x, model)
        end
        save_density_image(
            Array(x),
            "iteration_$(lpad(iteration, 2, '0')).png",
            "Iteration $iteration",
            controls,
        )
        println(
            "Iteration $iteration: loss = $(Reactant.to_number(loss)), " *
            "gradient = $gradient_norm, λ = $λₖ",
        )

        if iteration < controls.maximum_iterations
            Δxₖ = norm(Array(x - previous_x))
            Δ∇fₖ = norm(Array(gradient - previous_gradient))
            λₖ₋₁, θₖ₋₁ = λₖ, θₖ
            λₖ = min(
                sqrt(1 + θₖ₋₁) * λₖ₋₁,
                iszero(Δ∇fₖ) ? λₖ₋₁ : Δxₖ / (2Δ∇fₖ),
            )
            θₖ = λₖ / λₖ₋₁
        end
    end

    density = Array(x)
    save_density_image(density, "reconstructed_density.png", "Reconstructed brain density", controls)
    println("Final loss: ", Reactant.to_number(loss))
    println("Density extrema: ", extrema(density))
    println("Saved images to: ", controls.iteration_directory)
    return x
end

function run_ad_reconstruction(voxel_grid; navigator_correction)
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
    iteration_directory = joinpath(@__DIR__, "ADDiff$(resolution)x$(resolution)")
    mkpath(iteration_directory)

    recon_size = (128, 128)
    subspin_grid = (5, 2, 1)
    simulation_spins_per_voxel = prod(subspin_grid)
    navigator_count = 3
    brain_T1 = 1.0
    brain_T2 = 0.1
    brain_T2s = 0.05
    maximum_iterations = 20
    λ₀ = 1f-6

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
    measured_image = ComplexF32.(reduce(vcat, profile.data for profile in image_profiles))
    if navigator_correction
        navigator_profiles = raw_measured.profiles[1:navigator_count]
        measured_navigators =
            ComplexF32.(reduce(vcat, profile.data for profile in navigator_profiles))
        b = correct_navigator_phase(measured_navigators, measured_image, image_profiles)
        image_sample_offset = length(axes(measured_navigators, 1))
    else
        b = measured_image
        image_sample_offset = navigator_count * size(first(image_profiles).data, 1)
    end
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
    physio = CardiacSignal(; heart_rate=1)
    sim_params = Dict{String,Any}(
        "gpu" => true,
        "precision" => "f32",
        "sim_method" => Bloch(),
        "return_type" => "mat",
    )
    stencil = bilinear_stencil(voxel_axis_x, voxel_axis_y, spin_x, spin_y)
    stencil = merge(stencil, (; wx=Float32.(stencil.wx), wy=Float32.(stencil.wy)))
    receiver_values = ComplexF32.(get_sens(receiver, spin_x, spin_y, spin_z))
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
        maximum_iterations,
        λ₀,
        iteration_directory,
        object_name,
        stencil,
        simulation_grid,
        fov,
        voxel_count,
    )

    fft_model = fft_encoding(params, receiver_values)

    probe = Float32.(range(0.1, 1; length=voxel_count))
    reference_signal = Array(koma_signal(probe, params))
    fft_probe = fft_signal(ComplexF32.(probe), fft_model)
    relative_forward_error = norm(fft_probe - reference_signal) /
                             max(norm(reference_signal), eps(Float32))
    println("FFT forward relative error: ", relative_forward_error)
    relative_forward_error < 5e-4 || error("FFT forward does not match KomaMRI")

    forward_inner_product = dot(fft_probe, b)
    adjoint_inner_product = dot(ComplexF32.(probe), fft_adjoint(b, fft_model))
    relative_adjoint_error = abs(forward_inner_product - adjoint_inner_product) /
                             max(abs(forward_inner_product), eps(Float32))
    println("FFT adjoint relative error: ", relative_adjoint_error)
    relative_adjoint_error < 5e-5 || error("FFT adjoint does not match its forward")

    matched_filter = fft_adjoint(b, fft_model)
    matched_signal = fft_signal(matched_filter, fft_model)
    step = real(dot(matched_signal, b)) / sum(abs2, matched_signal)
    complex_density = ComplexF32.(step .* matched_filter)
    density_scale = maximum(abs, complex_density)
    density_scale > eps(Float32) || error("Matched-filter calibration is zero")
    spatial_phase = complex_density ./ max.(abs.(complex_density), eps(Float32))
    save_density_image(
        abs.(complex_density) ./ density_scale,
        "phase_calibration.png",
        "Matched-filter spatial-phase calibration",
        params,
    )

    controls = (; voxel_grid, maximum_iterations, λ₀, iteration_directory)
    model_device = (;
        sample_indices=Reactant.to_rarray(fft_model.sample_indices),
        sample_weight=Reactant.to_rarray(fft_model.sample_weight),
        receiver_values=Reactant.to_rarray(fft_model.receiver_values),
        stencil=map(Reactant.to_rarray, fft_model.stencil),
        simulation_grid=fft_model.simulation_grid,
        simulation_spins_per_voxel=fft_model.simulation_spins_per_voxel,
        b=Reactant.to_rarray(b ./ density_scale),
        spatial_phase=Reactant.to_rarray(spatial_phase),
    )
    x = Reactant.to_rarray(zeros(Float32, voxel_count))
    compiled_gradient = Reactant.@compile sync=true f_gradient(x, model_device)
    compiled_loss = Reactant.@compile sync=true f(x, model_device)
    initial_state = compiled_gradient(x, model_device)
    analytic_gradient = -2 .* real.(
        conj.(spatial_phase) .* fft_adjoint(b ./ density_scale, fft_model),
    )
    relative_gradient_error = norm(Array(last(initial_state)) - analytic_gradient) /
                              max(norm(analytic_gradient), eps(Float32))
    println("AD gradient relative error: ", relative_gradient_error)
    relative_gradient_error < 5e-4 || error("AD gradient does not match FFT adjoint")
    return reconstruct(
        x,
        controls,
        model_device,
        compiled_gradient,
        compiled_loss,
        initial_state,
    )
end
