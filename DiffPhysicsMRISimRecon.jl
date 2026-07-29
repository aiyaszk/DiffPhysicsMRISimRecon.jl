# Differentiable MRI simulation for physics-based reconstruction
# Density and T1 reconstruction using KomaMRI as the forward model

# Activate environment
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# MRI simulation
using KomaMRI
# Optimization
using Optim

include(joinpath(@__DIR__, "reconstruction.jl"))

########################################################################
# Reconstruction setup
########################################################################

quick = "--quick" in ARGS
density_iterations = quick ? 2 : 30
t1_iterations = quick ? 2 : 40

experiment = make_experiment()
full_epi = epi_sequence(experiment)
accelerated_epi = accelerate_epi(full_epi)

density_truth = log.(experiment.density)
density_initial = fill(log(0.5), length(density_truth))

########################################################################
# Fully sampled density reconstruction
########################################################################

println("#################### Fully sampled density ####################")

full_forward = parameters ->
    signal(experiment, exp.(parameters), experiment.t1, full_epi)
full_data = full_forward(density_truth)
full_result = reconstruct(
    full_forward,
    full_data,
    density_initial;
    iterations=density_iterations,
)
full_density = exp.(Optim.minimizer(full_result))

println("loss = ", Optim.minimum(full_result))
println("relative error = ", relative_error(full_density, experiment.density))

########################################################################
# R=2 accelerated density reconstruction
########################################################################

println("#################### R=2 density ####################")

accelerated_forward = parameters ->
    signal(experiment, exp.(parameters), experiment.t1, accelerated_epi)
accelerated_data = accelerated_forward(density_truth)
accelerated_result = reconstruct(
    accelerated_forward,
    accelerated_data,
    density_initial;
    iterations=density_iterations,
)
accelerated_density = exp.(Optim.minimizer(accelerated_result))

println("loss = ", Optim.minimum(accelerated_result))
println("relative error = ", relative_error(accelerated_density, experiment.density))

########################################################################
# Joint density and T1 reconstruction
########################################################################

println("#################### R=2 density + T1 ####################")

inversion_times = (0.08, 0.35, 1.0)
inversion_recovery_sequences = map(inversion_times) do inversion_time
    accelerate_epi(inversion_recovery_epi(experiment, inversion_time))
end

spins = length(experiment.phantom)
joint_forward = function(parameters)
    density = exp.(@view parameters[1:spins])
    t1 = exp.(@view parameters[(spins + 1):end])
    return reduce(
        vcat,
        (
            signal(experiment, density, t1, sequence) for
            sequence in inversion_recovery_sequences
        ),
    )
end

joint_truth = [log.(experiment.density); log.(experiment.t1)]
joint_data = joint_forward(joint_truth)
joint_initial = [fill(log(0.5), spins); fill(log(0.9), spins)]
joint_result = reconstruct(
    joint_forward,
    joint_data,
    joint_initial;
    iterations=t1_iterations,
)

joint_estimate = Optim.minimizer(joint_result)
joint_density = exp.(@view joint_estimate[1:spins])
joint_t1 = exp.(@view joint_estimate[(spins + 1):end])

println("loss = ", Optim.minimum(joint_result))
println("density error = ", relative_error(joint_density, experiment.density))
println("T1 error = ", relative_error(joint_t1, experiment.t1))
println("Finished!")
