# Finite-difference recovery of a cross density image

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using FiniteDiff, KomaMRI
using Interpolations: Flat, linear_interpolation
using KomaMRIBase: Phantom
using LinearAlgebra: norm
@eval using $(Sys.isapple() ? :Metal : :CUDA)

archive_directory = isempty(ARGS) ? joinpath(homedir(), "Desktop/Archive (1)") : first(ARGS)
sequence_file = joinpath(archive_directory, "seq/hard_epi_20interleaves_1avg_fatsat.seq")
iteration_directory = joinpath(@__DIR__, "DiffCrossTestResults")
mkpath(iteration_directory)

voxel_grid = (6, 6, 1)
subspin_grid = (5, 2, 1)
simulation_spins_per_voxel = prod(subspin_grid)
finite_difference_step = cbrt(eps(Float32))
maximum_iterations = 20

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

function simulation_object(x)
    interpolation = linear_interpolation(
        (voxel_axis_x, voxel_axis_y), reshape(x, voxel_grid[1:2]);
        extrapolation_bc=Flat(),
    )
    Phantom(;
        name="cross test simulation spins",
        x=spin_x,
        y=spin_y,
        z=spin_z,
        ρ=interpolation.(spin_x, spin_y) ./ simulation_spins_per_voxel,
        T1=fill(Inf, total_spin_count),
        T2=fill(Inf, total_spin_count),
        T2s=fill(Inf, total_spin_count),
    )
end

sim_params = Dict{String,Any}("gpu" => true, "precision" => "f32", "sim_method" => Bloch(), "return_type" => "mat")

A(x) = simulate(
    simulation_object(x), seq, scanner;
    sim_params, physio=CardiacSignal(; heart_rate=1), verbose=false,
)

x_true = vec(Float64[0 0 0 0 0 0; 0 0 1 1 0 0; 0 1 1 1 1 0; 0 1 1 1 1 0; 0 0 1 1 0 0; 0 0 0 0 0 0])
b = A(x_true)
f(x) = sum(abs2, A(x) - b) # Evaluates ‖A(x) - b‖²

save_density_image(x, name, title) = savefig(
    plot_image(reshape(x, voxel_grid[1:2]); title), joinpath(iteration_directory, name),
)

save_density_image(x_true, "truth.png", "True cross density")
x = zeros(voxel_count)
save_density_image(x, "iteration_00.png", "Iteration 0")

for iteration in 1:maximum_iterations # Repeats gradient descent up to the iteration limit.
    loss = f(x) # Records the current loss for step acceptance.
    gradient = FiniteDiff.finite_difference_gradient( # Estimates every density derivative.
        f, x; relstep=finite_difference_step, # Uses central differences with the Float32 step.
    )
    gradient_norm = norm(gradient) # Measures the gradient size for reporting.
    iszero(gradient_norm) && break # Stops before division when the gradient vanishes.
    step = 1 / gradient_norm # Proposes a scale-independent unit update.
    while f(x .- step .* gradient) > loss # Rejects updates that increase the loss.
        step /= 2 # Halves the update until the loss decreases.
    end
    x .-= step .* gradient # Applies the accepted update.

    save_density_image(x, "iteration_$(lpad(iteration, 2, '0')).png", "Iteration $iteration")
    println("Iteration $iteration: loss = $(f(x)), gradient = $gradient_norm")
end

save_density_image(x, "reconstructed_density.png", "Reconstructed cross density")
println("Final loss: ", f(x))
println("Saved images to: ", iteration_directory)
