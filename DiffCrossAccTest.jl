# Finite-difference recovery of a cross from an accelerated acquisition

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using FiniteDiff, KomaMRI
using Interpolations: Flat, linear_interpolation
using KomaMRIBase: Phantom
using LinearAlgebra: norm
@eval using $(Sys.isapple() ? :Metal : :CUDA)

archive_directory = isempty(ARGS) ? joinpath(homedir(), "Desktop/Archive (1)") : first(ARGS)
sequence_file = joinpath(archive_directory, "seq/hard_epi_2x_20interleaves_5avg_fatsat.seq")
iteration_directory = joinpath(@__DIR__, "DiffCrossAccTestResults")
mkpath(iteration_directory)

voxel_grid = (6, 6, 1)
subspin_grid = (5, 2, 1)
simulation_spins_per_voxel = prod(subspin_grid)
finite_difference_step = cbrt(eps(Float32))
maximum_iterations = 20
λ₀ = 1e-6 # Uses the default from the adaptive-gradient reference implementation.

seq = read_seq(sequence_file)
fov = Float64.(seq.DEF["FOV"])
scanner = Scanner(; receiver=BirdcageCoilSens(; ncoils=8))

centered_axis(width, count) = range(-width / 2 + width / (2count); step=width / count, length=count) # Places samples at cell centers.
voxel_count = prod(voxel_grid) # Sets the 36 reconstructed densities.
voxel_axis_x, voxel_axis_y = centered_axis.(fov[1:2], voxel_grid[1:2]) # Locates densities for interpolation.
simulation_grid = voxel_grid .* subspin_grid # Expands each voxel into ten simulation spins.
simulation_axes = centered_axis.(fov, simulation_grid) # Creates the simulation-spin coordinate axes.
spin_points = vec(collect(Iterators.product(simulation_axes...))) # Flattens the 3D grid for Phantom.
spin_x = first.(spin_points) # Gives Phantom every spin's x position.
spin_y = getindex.(spin_points, 2) # Gives Phantom every spin's y position.
spin_z = last.(spin_points) # Gives Phantom every spin's z position.
total_spin_count = length(spin_points) # Counts all 36 × 10 spins for Phantom.

function simulation_object(x, params)
    interpolation = linear_interpolation(
        (params.voxel_axis_x, params.voxel_axis_y), reshape(x, params.voxel_grid[1:2]);
        extrapolation_bc=Flat(),
    )
    Phantom(;
        name="cross test simulation spins",
        x=params.spin_x,
        y=params.spin_y,
        z=params.spin_z,
        ρ=interpolation.(params.spin_x, params.spin_y) ./ params.simulation_spins_per_voxel,
        T1=fill(Inf, params.total_spin_count),
        T2=fill(Inf, params.total_spin_count),
        T2s=fill(Inf, params.total_spin_count),
    )
end

sim_params = Dict{String,Any}("gpu" => true, "precision" => "f32", "sim_method" => Bloch(), "return_type" => "mat")
physio = CardiacSignal(; heart_rate=1)
params = (;
    seq, scanner, sim_params, physio, voxel_grid, voxel_axis_x, voxel_axis_y,
    spin_x, spin_y, spin_z, total_spin_count, simulation_spins_per_voxel,
    finite_difference_step, maximum_iterations, λ₀, iteration_directory,
)

A(x, params) = simulate(
    simulation_object(x, params), params.seq, params.scanner;
    sim_params=params.sim_params, physio=params.physio, verbose=false,
)

x_true = vec(Float64[0 0 0 0 0 0; 0 0 1 1 0 0; 0 1 1 1 1 0; 0 1 1 1 1 0; 0 0 1 1 0 0; 0 0 0 0 0 0])
b = A(x_true, params)
params = (; params..., b)
f(x, params)::Float32 = sum(abs2, A(x, params) - params.b) # Evaluates ‖A(x) - b‖² with the configured f32 precision.

save_density_image(x, name, title, params) = savefig(
    plot_image(reshape(x, params.voxel_grid[1:2]); title), joinpath(params.iteration_directory, name),
)

save_density_image(x_true, "truth.png", "True cross density", params)

function reconstruct(x, params) # Receives the initial densities and every constant optimization parameter.
    objective(x) = f(x, params) # Fixes params while FiniteDiff varies x.
    save_density_image(x, "iteration_00.png", "Iteration 0", params)
    ∇fₖ = FiniteDiff.finite_difference_gradient(objective, x; relstep=params.finite_difference_step) # Computes ∇f(x⁰).
    xₖ₋₁ = copy(x) # Stores the previous x for the adaptive rule.
    ∇fₖ₋₁ = ∇fₖ # Stores the previous gradient for the adaptive rule.
    λₖ = params.λ₀ # Starts with the specified λ₀.
    θₖ = Inf # Sets θ₀ so the first adaptive step is limited only by local curvature.

    for iteration in 1:params.maximum_iterations # Repeats the adaptive update up to the iteration limit.
        if iteration > 1 # Adapts λₖ after two parameter-gradient pairs are available.
            ∇fₖ = FiniteDiff.finite_difference_gradient(objective, x; relstep=params.finite_difference_step) # Computes ∇f(xᵏ).
            Δxₖ = norm(x - xₖ₋₁) # Measures ‖xᵏ - xᵏ⁻¹‖.
            Δ∇fₖ = norm(∇fₖ - ∇fₖ₋₁) # Measures ‖∇f(xᵏ) - ∇f(xᵏ⁻¹)‖.
            λₖ₋₁, θₖ₋₁ = λₖ, θₖ # Keeps the previous λ and θ used by Algorithm 1.
            λₖ = min(sqrt(1 + θₖ₋₁) * λₖ₋₁, iszero(Δ∇fₖ) ? λₖ₋₁ : Δxₖ / (2Δ∇fₖ)) # Computes the adaptive step.
            θₖ = λₖ / λₖ₋₁ # Computes θₖ for the next iteration.
        end
        ∇fₖ_norm = norm(∇fₖ) # Measures the gradient size for stopping and reporting.
        iszero(∇fₖ_norm) && break # Stops when there is no descent direction.
        xₖ₋₁ .= x # Saves xᵏ before changing x.
        ∇fₖ₋₁ = ∇fₖ # Saves ∇f(xᵏ) before computing the next gradient.
        x .-= λₖ .* ∇fₖ # Computes xᵏ⁺¹ = xᵏ - λₖ∇f(xᵏ).

        save_density_image(x, "iteration_$(lpad(iteration, 2, '0')).png", "Iteration $iteration", params)
        println("Iteration $iteration: loss = $(objective(x)), gradient = $∇fₖ_norm, λ = $λₖ")
    end

    save_density_image(x, "reconstructed_density.png", "Reconstructed cross density", params)
    println("Final loss: ", objective(x))
    println("Saved images to: ", params.iteration_directory)
    x
end

reconstruct(zeros(voxel_count), params)
