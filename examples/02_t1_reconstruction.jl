using DifferentiableKomaMRI
using Optim

include("common.jl")

function main()
    experiment_data = experiment()
    inversion_times = (0.08, 0.35, 1.0)
    sequences = map(inversion_times) do inversion_time
        full_sequence = inversion_recovery_epi(
            experiment_data.field_of_view,
            experiment_data.matrix_size,
            inversion_time,
            experiment_data.scanner,
        )
        accelerate_epi(full_sequence, 2)
    end
    problem = t1_problem(experiment_data, sequences)
    methods = ("gradient descent" => GradientDescent(), "L-BFGS" => LBFGS())
    iterations = "--quick" in ARGS ? 2 : 40
    options = Optim.Options(; iterations, g_tol=1e-8)

    for (name, method) in methods
        result = reconstruct(
            problem.forward,
            problem.data,
            problem.initial,
            method;
            options,
        )
        parameters = Optim.minimizer(result)
        spins = length(experiment_data.template)
        density = exp.(@view parameters[1:spins])
        t1 = exp.(@view parameters[(spins + 1):end])
        forward_evaluations =
            Optim.f_calls(result) +
            2length(problem.initial) * Optim.g_calls(result)
        println(
            rpad(name, 16),
            " loss=", Optim.minimum(result),
            " density_error=", relative_error(density, experiment_data.density),
            " T1_error=", relative_error(t1, experiment_data.t1),
            " iterations=", Optim.iterations(result),
            " forward_evaluations=", forward_evaluations,
        )
    end
    return nothing
end

main()
