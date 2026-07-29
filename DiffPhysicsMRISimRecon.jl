using Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using KomaMRI
using Optim

include(joinpath(@__DIR__, "reconstruction.jl"))

function run_density(experiment, sequence, name, iterations)
    forward(parameters) = signal(experiment, exp.(parameters), experiment.t1, sequence)
    truth = log.(experiment.density)
    data = forward(truth)
    initial = fill(log(0.5), length(truth))
    result = reconstruct(forward, data, initial; iterations)
    estimate = exp.(Optim.minimizer(result))
    println(
        rpad(name, 18),
        "loss=", Optim.minimum(result),
        "  error=", relative_error(estimate, experiment.density),
    )
end

function run_t1(experiment, iterations)
    sequences = map((0.08, 0.35, 1.0)) do inversion_time
        accelerate_epi(inversion_recovery_epi(experiment, inversion_time))
    end
    spins = length(experiment.phantom)
    function forward(parameters)
        density = exp.(@view parameters[1:spins])
        t1 = exp.(@view parameters[(spins + 1):end])
        return reduce(vcat, (signal(experiment, density, t1, sequence) for sequence in sequences))
    end

    truth = [log.(experiment.density); log.(experiment.t1)]
    data = forward(truth)
    initial = [fill(log(0.5), spins); fill(log(0.9), spins)]
    result = reconstruct(forward, data, initial; iterations)
    estimate = Optim.minimizer(result)
    density = exp.(@view estimate[1:spins])
    t1 = exp.(@view estimate[(spins + 1):end])
    println(
        rpad("density + T1 R=2", 18),
        "loss=", Optim.minimum(result),
        "  density_error=", relative_error(density, experiment.density),
        "  T1_error=", relative_error(t1, experiment.t1),
    )
end

function main()
    quick = "--quick" in ARGS
    experiment = make_experiment()
    full_epi = epi_sequence(experiment)

    println("Density reconstruction")
    run_density(experiment, full_epi, "fully sampled", quick ? 2 : 30)
    run_density(experiment, accelerate_epi(full_epi), "R=2", quick ? 2 : 30)

    println("\nJoint quantitative reconstruction")
    run_t1(experiment, quick ? 2 : 40)
end

main()
