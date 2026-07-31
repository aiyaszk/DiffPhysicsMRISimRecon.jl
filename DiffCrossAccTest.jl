# Finite-difference recovery of a cross from an accelerated acquisition

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using FiniteDiff, KomaMRI
using Interpolations: Flat, linear_interpolation
using KomaMRIBase: Phantom
using LinearAlgebra: norm

if Sys.isapple()
    @eval using Metal
else
    @eval using CUDA
end

archive_directory = isempty(ARGS) ? joinpath(homedir(), "Desktop/Archive (1)") : first(ARGS)
sequence_file = joinpath(archive_directory, "seq/hard_epi_2x_20interleaves_5avg_fatsat.seq")
iteration_directory = joinpath(@__DIR__, "DiffCrossAccTestResults")
mkpath(iteration_directory)

voxel_grid = (6, 6)
subspin_grid = (5, 2)
spins_per_voxel_per_slice = prod(subspin_grid)
slice_count = 10
simulation_spins_per_voxel = spins_per_voxel_per_slice * slice_count
finite_difference_step = cbrt(eps(Float32))
maximum_iterations = 20

seq = read_seq(sequence_file)
fov = Float64.(seq.DEF["FOV"])
scanner = Scanner(; receiver=BirdcageCoilSens(; ncoils=8))

voxel_spacing_x = fov[1] / voxel_grid[1]
voxel_spacing_y = fov[2] / voxel_grid[2]
voxel_axis_x = collect(
    range(
        -fov[1] / 2 + voxel_spacing_x / 2,
        fov[1] / 2 - voxel_spacing_x / 2;
        length=voxel_grid[1],
    ),
)
voxel_axis_y = collect(
    range(
        -fov[2] / 2 + voxel_spacing_y / 2,
        fov[2] / 2 - voxel_spacing_y / 2;
        length=voxel_grid[2],
    ),
)
voxel_x = repeat(voxel_axis_x, voxel_grid[2])
voxel_y = repeat(voxel_axis_y; inner=voxel_grid[1])
voxel_count = length(voxel_x)

simulation_grid = voxel_grid .* subspin_grid
simulation_spacing_x = fov[1] / simulation_grid[1]
simulation_spacing_y = fov[2] / simulation_grid[2]
simulation_axis_x = collect(
    range(
        -fov[1] / 2 + simulation_spacing_x / 2,
        fov[1] / 2 - simulation_spacing_x / 2;
        length=simulation_grid[1],
    ),
)
simulation_axis_y = collect(
    range(
        -fov[2] / 2 + simulation_spacing_y / 2,
        fov[2] / 2 - simulation_spacing_y / 2;
        length=simulation_grid[2],
    ),
)
slice_spacing = fov[3] / slice_count
slice_axis = collect(
    range(-fov[3] / 2 + slice_spacing / 2, fov[3] / 2 - slice_spacing / 2; length=slice_count),
)
inplane_spin_x = repeat(simulation_axis_x, simulation_grid[2])
inplane_spin_y = repeat(simulation_axis_y; inner=simulation_grid[1])
spin_x = repeat(inplane_spin_x, slice_count)
spin_y = repeat(inplane_spin_y, slice_count)
spin_z = repeat(slice_axis; inner=length(inplane_spin_x))
spin_count = length(spin_x)

function simulation_object(density)
    interpolation = linear_interpolation(
        (voxel_axis_x, voxel_axis_y),
        reshape(density, voxel_grid);
        extrapolation_bc=Flat(),
    )
    Phantom(;
        name="cross test simulation spins",
        x=spin_x,
        y=spin_y,
        z=spin_z,
        ρ=interpolation.(spin_x, spin_y) ./ simulation_spins_per_voxel,
        T1=fill(Inf, spin_count),
        T2=fill(Inf, spin_count),
        T2s=fill(Inf, spin_count),
    )
end

sim_params = Dict{String,Any}(
    "gpu" => true,
    "precision" => "f32",
    "sim_method" => Bloch(),
    "return_type" => "mat",
)

forward(density) = simulate(
    simulation_object(density),
    seq,
    scanner;
    sim_params,
    physio=CardiacSignal(; heart_rate=1),
    verbose=false,
)

inside_vertical_arm =
    (abs.(voxel_x) .<= voxel_spacing_x) .& (abs.(voxel_y) .<= 2voxel_spacing_y)
inside_horizontal_arm =
    (abs.(voxel_x) .<= 2voxel_spacing_x) .& (abs.(voxel_y) .<= voxel_spacing_y)
true_density = Float64.(inside_vertical_arm .| inside_horizontal_arm)
target = forward(true_density)

objective(density) = begin
    residual = forward(density) - target
    sum(abs2, residual) / (2sum(abs2, target))
end

function save_density_image(density, name, title)
    savefig(plot_image(reshape(density, voxel_grid); title), joinpath(iteration_directory, name))
end

save_density_image(true_density, "truth.png", "True cross density")
parameters = zeros(voxel_count)
save_density_image(parameters, "iteration_00.png", "Iteration 0")

for iteration in 1:maximum_iterations
    loss = objective(parameters)
    local gradient = FiniteDiff.finite_difference_gradient(
        objective,
        parameters;
        relstep=finite_difference_step,
        absstep=finite_difference_step,
    )
    gradient_norm = norm(gradient)
    gradient_norm <= 1e-8 && break

    step = inv(gradient_norm)
    while objective(parameters .- step .* gradient) > loss
        step /= 2
    end
    parameters .-= step .* gradient
    save_density_image(
        parameters,
        "iteration_$(lpad(iteration, 2, '0')).png",
        "Iteration $iteration",
    )
    println("Iteration $iteration: loss = $(objective(parameters)), gradient = $gradient_norm")
end

reconstructed_density = parameters
save_density_image(reconstructed_density, "reconstructed_density.png", "Reconstructed cross density")
println("Final loss: ", objective(parameters))
println("Saved iteration images to: ", iteration_directory)
