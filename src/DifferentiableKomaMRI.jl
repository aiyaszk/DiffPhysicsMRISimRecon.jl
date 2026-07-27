module DifferentiableKomaMRI

import Optim

export finite_difference_gradient, finite_difference_gradient!, least_squares_objective
export reconstruct

"""
    finite_difference_gradient!(gradient, objective, parameters; relative_step)

Calculate the gradient of a real scalar objective using central finite differences.
"""
function finite_difference_gradient!(
    gradient,
    objective,
    parameters;
    relative_step=cbrt(eps(float(one(eltype(parameters))))),
)
    for parameter in eachindex(parameters)
        step = relative_step * max(abs(parameters[parameter]), one(eltype(parameters)))
        positive = copy(parameters)
        negative = copy(parameters)
        positive[parameter] += step
        negative[parameter] -= step
        gradient[parameter] = (objective(positive) - objective(negative)) / (2step)
    end
    return gradient
end

"""
    finite_difference_gradient(objective, parameters; relative_step)

Allocate and return a central finite-difference gradient.
"""
function finite_difference_gradient(objective, parameters; kwargs...)
    gradient = similar(parameters)
    return finite_difference_gradient!(gradient, objective, parameters; kwargs...)
end

"""
    least_squares_objective(forward, data)

Construct the real least-squares objective `parameters -> ‖forward(parameters) - data‖² / 2`.
The forward model and data may be complex-valued.
"""
least_squares_objective(forward, data) =
    parameters -> sum(abs2, forward(parameters) - data) / 2

"""
    reconstruct(forward, data, initial_parameters, method; options, relative_step)

Minimize a complex least-squares data-fidelity objective using an `Optim` method and a
central finite-difference gradient.
"""
function reconstruct(
    forward,
    data,
    initial_parameters,
    method;
    options=Optim.Options(),
    relative_step=cbrt(eps(float(one(eltype(initial_parameters))))),
)
    objective = least_squares_objective(forward, data)
    gradient!(storage, parameters) = finite_difference_gradient!(
        storage,
        objective,
        parameters;
        relative_step,
    )
    return Optim.optimize(
        objective,
        gradient!,
        collect(float.(initial_parameters)),
        method,
        options,
    )
end

end
