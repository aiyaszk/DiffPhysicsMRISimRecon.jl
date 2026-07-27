using KomaMRI

const SIMULATION_PARAMETERS = Dict{String,Any}(
    "gpu" => false,
    "Nthreads" => 1,
    "return_type" => "mat",
    "precision" => "f64",
    "sim_method" => BlochSimple(),
)

relative_error(estimate, truth) = sqrt(sum(abs2, estimate - truth) / sum(abs2, truth))

function epi_sequence(field_of_view, matrix_size, scanner)
    excitation = PulseDesigner.build_block_pulse(π / 2; duration=1e-3, sys=scanner)
    return excitation + PulseDesigner.EPI(field_of_view, matrix_size, scanner)
end

function accelerate_epi(sequence, acceleration)
    readout_blocks = findall(block -> is_ADC_on(sequence[block]), 1:length(sequence))
    length(readout_blocks) % acceleration == 0 ||
        error("Readout count must be divisible by the acceleration")

    last_readout = readout_blocks[length(readout_blocks) ÷ acceleration]
    trailing_blocks = (last(readout_blocks) + 1):length(sequence)
    accelerated = deepcopy(sequence[1:last_readout] + sequence[trailing_blocks])
    phase_blips = (first(readout_blocks) + 1):2:(last_readout - 1)
    accelerated.GR[2, phase_blips] .= acceleration .* accelerated.GR[2, phase_blips]

    _, full_kspace = get_kspace(sequence)
    _, accelerated_kspace = get_kspace(accelerated)
    full_center = argmin(vec(sum(abs2, full_kspace; dims=2)))
    accelerated_center = argmin(vec(sum(abs2, accelerated_kspace; dims=2)))
    echo_delay =
        get_adc_sampling_times(sequence)[full_center] -
        get_adc_sampling_times(accelerated)[accelerated_center]
    echo_delay >= 0 || error("Accelerated sequence centre occurs after the full sequence")
    return accelerated[1:(first(readout_blocks) - 1)] +
           Delay(echo_delay) +
           accelerated[first(readout_blocks):end]
end

function inversion_recovery_epi(field_of_view, matrix_size, inversion_time, scanner)
    inversion = PulseDesigner.build_block_pulse(π; duration=1e-3, sys=scanner)
    excitation = PulseDesigner.build_block_pulse(π / 2; duration=1e-3, sys=scanner)
    recovery_delay = inversion_time - (dur(inversion) + dur(excitation)) / 2
    recovery_delay > 0 || error("Inversion time is too short for the RF pulses")
    readout = PulseDesigner.EPI(field_of_view, matrix_size, scanner)
    return inversion + Delay(recovery_delay) + excitation + readout
end

function experiment()
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
    template = Phantom(; x, y, ρ=density, T1=t1, T2=fill(0.08, length(x)))
    return (; field_of_view, matrix_size, scanner, template, density, t1)
end

function signal(experiment, density, t1, sequence)
    object = copy(experiment.template)
    object.ρ = density
    object.T1 = t1
    return vec(simulate(
        object,
        sequence,
        experiment.scanner;
        sim_params=SIMULATION_PARAMETERS,
        verbose=false,
    ))
end

function density_problem(experiment, sequence)
    forward(parameters) = signal(experiment, exp.(parameters), experiment.t1, sequence)
    truth = log.(experiment.density)
    data = forward(truth)
    initial = fill(log(0.5), length(truth))
    return (; forward, data, initial, truth)
end

function t1_problem(experiment, sequences)
    spins = length(experiment.template)
    function forward(parameters)
        density = exp.(@view parameters[1:spins])
        t1 = exp.(@view parameters[(spins + 1):end])
        return reduce(
            vcat,
            (signal(experiment, density, t1, sequence) for sequence in sequences),
        )
    end
    truth = [log.(experiment.density); log.(experiment.t1)]
    data = forward(truth)
    initial = [fill(log(0.5), spins); fill(log(0.9), spins)]
    return (; forward, data, initial, truth)
end
