# Finite-difference reconstruction on a 64 × 64 voxel grid

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using FiniteDiff, KomaMRI, MRICoilSensitivities
using Interpolations: Flat, linear_interpolation
using KomaMRIBase: ArbitraryCoilSens, Phantom
using LinearAlgebra: norm
using Statistics: mean

if Sys.isapple()
    @eval using Metal
else
    @eval using CUDA
end

# Local acquisition files
archive_directory = isempty(ARGS) ? joinpath(homedir(), "Desktop/Archive (1)") : first(ARGS)
fully_sampled_mrd_file = joinpath(
    archive_directory,
    "mrd_hdf5/meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd",
)
accelerated_2x_mrd_file = joinpath(
    archive_directory,
    "mrd_hdf5/meas_MID01109_FID34203_hard_epi_2x_20interleaves_5avg_fatsat.mrd",
)
accelerated_2x_seq_file = joinpath(
    archive_directory,
    "seq/hard_epi_2x_20interleaves_5avg_fatsat.seq",
)
simulated_mrd_file = joinpath(@__DIR__, "simulated_acquisition.mrd")

recon_size = (128, 128)
voxel_grid = (64, 64)
subspin_grid = (5, 2)
spins_per_voxel_per_slice = prod(subspin_grid)
slice_count = 10
simulation_spins_per_voxel = spins_per_voxel_per_slice * slice_count
navigators = 3
fixed_T1 = Inf
fixed_T2 = Inf
finite_difference_step = cbrt(eps(Float32))
maximum_iterations = 10
iteration_directory = joinpath(@__DIR__, "DiffPhysicsMRISimRecoIterations")
mkpath(iteration_directory)

# Sequence and acquired R=2 profiles
seq = read_seq(accelerated_2x_seq_file)
shots = Int(seq.DEF["EpiShots"])
acquired_shots = parse.(Int, split(seq.DEF["EpiAcquiredShots"], ',')) .- 1
acquired_lines = [
    line for shot in acquired_shots for line in shot:shots:(recon_size[2] - 1)
]

raw_measured = RawAcquisitionData(ISMRMRDFile(accelerated_2x_mrd_file))
image_profiles =
    raw_measured.profiles[(navigators + 1):(navigators + length(acquired_lines))]
measured_data = ComplexF64.(reduce(vcat, profile.data for profile in image_profiles))

adc_blocks = findall(block -> is_ADC_on(seq[block]), 1:length(seq))
seq = seq[1:adc_blocks[navigators + length(acquired_lines)]]

# ESPIRiT receiver maps
raw_reference = RawAcquisitionData(ISMRMRDFile(fully_sampled_mrd_file))
raw_reference.profiles = raw_reference.profiles[(navigators + 1):(navigators + recon_size[2])]
acq_reference = AcquisitionData(raw_reference)
acq_reference.traj[1].circular = false
sensitivity_maps = espirit(
    acq_reference,
    (6, 6),
    30,
    recon_size;
    eigThresh_1=0.02,
    eigThresh_2=0.0,
)

fov = Float64.(raw_reference.params["reconFOV"]) .* 1e-3
map_x = collect(LinRange(-fov[1] / 2, fov[1] / 2, recon_size[1]))
map_y = collect(LinRange(-fov[2] / 2, fov[2] / 2, recon_size[2]))
map_z = [-fov[3] / 2, 0.0, fov[3] / 2]
receiver = ArbitraryCoilSens(
    map_x,
    map_y,
    map_z,
    repeat(sensitivity_maps, 1, 1, length(map_z), 1),
)
scanner = Scanner(; receiver)

# Uniform reconstruction voxels
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
voxel_count = prod(voxel_grid)

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
        name="$(voxel_grid[1])x$(voxel_grid[2]) reconstruction voxels",
        x=spin_x,
        y=spin_y,
        z=spin_z,
        ρ=interpolation.(spin_x, spin_y) ./ simulation_spins_per_voxel,
        T1=fill(fixed_T1, spin_count),
        T2=fill(fixed_T2, spin_count),
        T2s=fill(fixed_T2, spin_count),
    )
end

sim_params = Dict{String,Any}(
    "gpu" => true,
    "precision" => "f32",
    "sim_method" => Bloch(),
    "return_type" => "mat",
)

function forward(density; return_type="mat")
    parameters = copy(sim_params)
    parameters["return_type"] = return_type
    simulate(
        simulation_object(density),
        seq,
        scanner;
        sim_params=parameters,
        physio=CardiacSignal(; heart_rate=1),
        verbose=false,
    )
end

voxel_density(parameters) = begin
    density = exp.(parameters .- maximum(parameters))
    density ./ mean(density)
end

function objective(parameters)
    signal = forward(voxel_density(parameters))
    predicted = signal[(end - size(measured_data, 1) + 1):end, :]
    coil_gain =
        sum(conj.(predicted) .* measured_data; dims=1) ./ sum(abs2, predicted; dims=1)
    residual = predicted .* coil_gain - measured_data
    sum(abs2, residual) / (2sum(abs2, measured_data))
end

function save_iteration_image(parameters, iteration)
    density = reshape(voxel_density(parameters), voxel_grid)
    file = joinpath(iteration_directory, "iteration_$(lpad(iteration, 2, '0')).png")
    savefig(plot_image(density; title="Iteration $iteration voxel density"), file)
end

parameters = zeros(voxel_count)
save_iteration_image(parameters, 0)
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
    save_iteration_image(parameters, iteration)
    println("Iteration $iteration: loss = $(objective(parameters)), gradient = $gradient_norm")
end

reconstructed_density = reshape(voxel_density(parameters), voxel_grid)
raw_simulated = forward(vec(reconstructed_density); return_type="raw")
save(ISMRMRDFile(simulated_mrd_file), raw_simulated)

density_image_file = joinpath(@__DIR__, "reconstructed_density.png")
savefig(
    plot_image(reconstructed_density; title="Finite-difference voxel density"),
    density_image_file,
)

println("Final loss: ", objective(parameters))
println("Saved: ", density_image_file)
println("Saved: ", simulated_mrd_file)
