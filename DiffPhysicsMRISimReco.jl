# Finite-difference reconstruction on a 126 × 126 node grid

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using FiniteDiff, KomaMRI, Metal, MRICoilSensitivities
using KomaMRIBase: ArbitraryCoilSens, Phantom
using LinearAlgebra: norm
using Statistics: mean

# Local acquisition files
fully_sampled_mrd_file = joinpath(
    homedir(),
    "Desktop/Archive (1)/mrd_hdf5/meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd",
)
accelerated_2x_mrd_file = joinpath(
    homedir(),
    "Desktop/Archive (1)/mrd_hdf5/meas_MID01109_FID34203_hard_epi_2x_20interleaves_5avg_fatsat.mrd",
)
accelerated_2x_seq_file = joinpath(
    homedir(),
    "Desktop/Archive (1)/seq/hard_epi_2x_20interleaves_5avg_fatsat.seq",
)
simulated_mrd_file = joinpath(@__DIR__, "simulated_acquisition.mrd")

recon_size = (128, 128)
node_grid = (126, 126)
navigators = 3
fixed_T1 = 1.0
fixed_T2 = 0.1
maximum_iterations = 10

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

# Uniform reconstruction nodes at the grid-cell centers
spacing_x = fov[1] / node_grid[1]
spacing_y = fov[2] / node_grid[2]
axis_x = collect(
    range(-fov[1] / 2 + spacing_x / 2, fov[1] / 2 - spacing_x / 2; length=node_grid[1]),
)
axis_y = collect(
    range(-fov[2] / 2 + spacing_y / 2, fov[2] / 2 - spacing_y / 2; length=node_grid[2]),
)
node_x = repeat(axis_x, node_grid[2])
node_y = repeat(axis_y; inner=node_grid[1])
node_count = length(node_x)

function node_object(density)
    Phantom(;
        name="$(node_grid[1])x$(node_grid[2]) reconstruction nodes",
        x=node_x,
        y=node_y,
        z=zeros(node_count),
        ρ=density,
        T1=fill(fixed_T1, node_count),
        T2=fill(fixed_T2, node_count),
        T2s=fill(fixed_T2, node_count),
    )
end

sim_params = Dict{String,Any}(
    "gpu" => true,
    "precision" => "f32",
    "sim_method" => Bloch(),
    "return_type" => "mat",
)

function simulate_nodes(density; return_type="mat")
    parameters = copy(sim_params)
    parameters["return_type"] = return_type
    simulate(
        node_object(density),
        seq,
        scanner;
        sim_params=parameters,
        physio=CardiacSignal(; heart_rate=1),
        verbose=false,
    )
end

node_density(parameters) = begin
    density = exp.(parameters .- maximum(parameters))
    density ./ mean(density)
end

function objective(parameters)
    signal = simulate_nodes(node_density(parameters))
    predicted = signal[(end - size(measured_data, 1) + 1):end, :]
    coil_gain =
        sum(conj.(predicted) .* measured_data; dims=1) ./ sum(abs2, predicted; dims=1)
    residual = predicted .* coil_gain - measured_data
    sum(abs2, residual) / (2sum(abs2, measured_data))
end

parameters = zeros(node_count)
for iteration in 1:maximum_iterations
    loss = objective(parameters)
    local gradient = FiniteDiff.finite_difference_gradient(objective, parameters)
    gradient_norm = norm(gradient)
    gradient_norm <= 1e-8 && break

    step = inv(gradient_norm)
    while objective(parameters .- step .* gradient) > loss
        step /= 2
    end
    parameters .-= step .* gradient
    println("Iteration $iteration: loss = $(objective(parameters)), gradient = $gradient_norm")
end

reconstructed_density = reshape(node_density(parameters), node_grid)
raw_simulated = simulate_nodes(vec(reconstructed_density); return_type="raw")
save(ISMRMRDFile(simulated_mrd_file), raw_simulated)

density_image_file = joinpath(@__DIR__, "reconstructed_density.png")
savefig(
    plot_image(reconstructed_density; title="Finite-difference node density"),
    density_image_file,
)

println("Final loss: ", objective(parameters))
println("Saved: ", density_image_file)
println("Saved: ", simulated_mrd_file)
