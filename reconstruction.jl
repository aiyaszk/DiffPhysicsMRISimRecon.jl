const SIMULATION_PARAMETERS = Dict{String,Any}(
    "gpu" => false,
    "Nthreads" => 1,
    "return_type" => "mat",
    "precision" => "f64",
    "sim_method" => BlochSimple(),
)

relative_error(estimate, truth) = sqrt(sum(abs2, estimate - truth) / sum(abs2, truth))

least_squares(forward, data) = parameters -> sum(abs2, forward(parameters) - data) / 2

function finite_difference_gradient!(gradient, objective, parameters)
    relative_step = cbrt(eps(eltype(parameters)))
    for index in eachindex(parameters)
        step = relative_step * max(abs(parameters[index]), one(eltype(parameters)))
        positive = copy(parameters)
        negative = copy(parameters)
        positive[index] += step
        negative[index] -= step
        gradient[index] = (objective(positive) - objective(negative)) / (2step)
    end
    return gradient
end

function reconstruct(forward, data, initial; iterations)
    objective = least_squares(forward, data)
    gradient!(storage, parameters) =
        finite_difference_gradient!(storage, objective, parameters)
    options = Optim.Options(; iterations, g_tol=1e-8)
    return Optim.optimize(objective, gradient!, float.(initial), Optim.LBFGS(), options)
end

function make_experiment()
    field_of_view = 0.23
    matrix_size = 4
    scanner = Scanner(; receiver=BirdcageCoilSens(; ncoils=4))
    pixel_width = field_of_view / matrix_size
    axis = range(
        -field_of_view / 2 + pixel_width / 2,
        field_of_view / 2 - pixel_width / 2;
        length=matrix_size,
    )
    x = vec([position for position in axis, _ in axis])
    y = vec([position for _ in axis, position in axis])
    foreground = Float64.(abs.(x) .+ abs.(y) .< field_of_view / 2)
    density = 0.15 .+ 0.85 .* foreground
    t1 = 0.7 .+ 0.5 .* foreground
    phantom = Phantom(; x, y, ρ=density, T1=t1, T2=fill(0.08, length(x)))
    return (; field_of_view, matrix_size, scanner, phantom, density, t1)
end

function epi_sequence(experiment)
    excitation = PulseDesigner.build_block_pulse(
        π / 2;
        duration=1e-3,
        sys=experiment.scanner,
    )
    return excitation + PulseDesigner.EPI(
        experiment.field_of_view,
        experiment.matrix_size,
        experiment.scanner,
    )
end

function accelerate_epi(sequence, acceleration=2)
    readouts = findall(block -> is_ADC_on(sequence[block]), 1:length(sequence))
    last_readout = readouts[length(readouts) ÷ acceleration]
    accelerated = deepcopy(sequence[1:last_readout] + sequence[(last(readouts) + 1):end])
    blips = (first(readouts) + 1):2:(last_readout - 1)
    accelerated.GR[2, blips] .*= acceleration

    _, full_kspace = get_kspace(sequence)
    _, accelerated_kspace = get_kspace(accelerated)
    full_center = argmin(vec(sum(abs2, full_kspace; dims=2)))
    accelerated_center = argmin(vec(sum(abs2, accelerated_kspace; dims=2)))
    echo_delay =
        get_adc_sampling_times(sequence)[full_center] -
        get_adc_sampling_times(accelerated)[accelerated_center]

    first_readout = first(readouts)
    return accelerated[1:(first_readout - 1)] +
           Delay(echo_delay) +
           accelerated[first_readout:end]
end

function inversion_recovery_epi(experiment, inversion_time)
    inversion = PulseDesigner.build_block_pulse(π; duration=1e-3, sys=experiment.scanner)
    excitation = PulseDesigner.build_block_pulse(
        π / 2;
        duration=1e-3,
        sys=experiment.scanner,
    )
    recovery_delay = inversion_time - (dur(inversion) + dur(excitation)) / 2
    readout = PulseDesigner.EPI(
        experiment.field_of_view,
        experiment.matrix_size,
        experiment.scanner,
    )
    return inversion + Delay(recovery_delay) + excitation + readout
end

function signal(experiment, density, t1, sequence)
    phantom = copy(experiment.phantom)
    phantom.ρ = density
    phantom.T1 = t1
    return vec(simulate(
        phantom,
        sequence,
        experiment.scanner;
        sim_params=SIMULATION_PARAMETERS,
        verbose=false,
    ))
end
