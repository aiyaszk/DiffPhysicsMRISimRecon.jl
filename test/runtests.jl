using DifferentiableKomaMRI
using Optim
using Test

@testset "Finite-difference gradients" begin
    objective(parameters) =
        (parameters[1] - 2)^2 + 3(parameters[2] + 1)^2 + parameters[1] * parameters[2]
    parameters = [0.3, -0.4]
    expected = [2(parameters[1] - 2) + parameters[2], 6(parameters[2] + 1) + parameters[1]]

    @test finite_difference_gradient(objective, parameters) ≈ expected rtol=1e-9
end

@testset "Complex least-squares reconstruction" begin
    forward(parameters) = ComplexF64[
        parameters[1] + im * parameters[2],
        2parameters[1] - im * parameters[2],
    ]
    truth = [0.7, -1.2]
    data = forward(truth)
    options = Optim.Options(; iterations=100, g_tol=1e-10)

    for method in (GradientDescent(), LBFGS())
        result = reconstruct(forward, data, zeros(2), method; options)
        @test Optim.converged(result)
        @test Optim.minimizer(result) ≈ truth atol=1e-7
    end
end
