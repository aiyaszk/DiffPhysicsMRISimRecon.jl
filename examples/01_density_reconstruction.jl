using DifferentiableKomaMRI
using Optim

include("common.jl")

function main()
    experiment_data = experiment()
    full_sequence = epi_sequence(
        experiment_data.field_of_view,
        experiment_data.matrix_size,
        experiment_data.scanner,
    )
    sequences = (
        "fully sampled" => full_sequence,
        "R=2" => accelerate_epi(full_sequence, 2),
    )
    methods = ("gradient descent" => GradientDescent(), "L-BFGS" => LBFGS())
    iterations = "--quick" in ARGS ? 2 : 30
    options = Optim.Options(; iterations, g_tol=1e-8)

    for (acquisition, sequence) in sequences
        problem = density_problem(experiment_data, sequence)
        println(acquisition)
        for (name, method) in methods
            result = reconstruct(
                problem.forward,
                problem.data,
                problem.initial,
                method;
                options,
            )
            estimate = exp.(Optim.minimizer(result))
            forward_evaluations =
                Optim.f_calls(result) +
                2length(problem.initial) * Optim.g_calls(result)
            println(
                "  ",
                rpad(name, 16),
                " loss=", Optim.minimum(result),
                " relative_error=", relative_error(estimate, experiment_data.density),
                " iterations=", Optim.iterations(result),
                " forward_evaluations=", forward_evaluations,
            )
        end
    end
    return nothing
end

main()
