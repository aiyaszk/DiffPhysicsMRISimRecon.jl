# Differentiable physics for MRI reconstruction

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using KomaMRI, MRICoilSensitivities
using KomaMRIBase: ArbitraryCoilSens
using LinearAlgebra: norm

# Local inputs
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
phantom_2D_3T_file = joinpath(
    homedir(),
    "Downloads/brain2D_3T_fat_z1cm_2x_xy.phantom",
)

# Acquisition and sensitivity maps
recon_size = (128, 128)
navigators = 3
seq = read_seq(accelerated_2x_seq_file)

shots = Int(seq.DEF["EpiShots"])
acquired_shots = parse.(Int, split(seq.DEF["EpiAcquiredShots"], ',')) .- 1
acquired_lines = [
    line for shot in acquired_shots for line in shot:shots:(recon_size[2] - 1)
]

raw_reference = RawAcquisitionData(ISMRMRDFile(fully_sampled_mrd_file))
raw_reference.profiles = raw_reference.profiles[(navigators + 1):(navigators + recon_size[2])]
acq_reference = AcquisitionData(raw_reference)
acq_reference.traj[1].circular = false
sensitivity_maps = espirit(acq_reference, (6, 6), 30, recon_size; eigThresh_1=0.02, eigThresh_2=0.0,)

fov = Float32.(raw_reference.params["reconFOV"]) .* 1f-3
x = collect(LinRange(-fov[1] / 2, fov[1] / 2, recon_size[1]))
y = collect(LinRange(-fov[2] / 2, fov[2] / 2, recon_size[2]))
z = Float32[-fov[3] / 2, 0, fov[3] / 2]
receiver = ArbitraryCoilSens(x, y, z, repeat(sensitivity_maps, 1, 1, length(z), 1),)
scanner = Scanner(; receiver)

raw_measured = RawAcquisitionData(ISMRMRDFile(accelerated_2x_mrd_file))
image_profiles =
    raw_measured.profiles[(navigators + 1):(navigators + length(acquired_lines))]
measured_data = reduce(vcat, profile.data for profile in image_profiles)

adc_blocks = findall(block -> is_ADC_on(seq[block]), 1:length(seq))
seq = seq[1:adc_blocks[navigators + length(acquired_lines)]]

# KomaMRI density forward model
phantom = read_phantom(phantom_2D_3T_file)
density_levels = sort(unique(phantom.ρ))
sim_params = Dict{String,Any}(
    "gpu" => false,
    "precision" => "f64",
    "sim_method" => Bloch(),
)

basis_signals = map(density_levels) do density
    indices = findall(==(density), phantom.ρ)
    count = min(50_000, length(indices))
    sampled = indices[round.(Int, range(1, length(indices); length=count))]
    component = phantom[sampled]
    component.ρ .= length(indices) / count

    println("Simulating density basis $density with $count spins")
    raw = simulate(
        component,
        seq,
        scanner;
        sim_params,
        physio=CardiacSignal(; heart_rate=1),
        verbose=false,
    )
    reduce(vcat, profile.data for profile in raw.profiles[(navigators + 1):end])
end
basis = cat(basis_signals...; dims=3)

nominal_signal = dropdims(
    sum(basis .* reshape(density_levels, 1, 1, :); dims=3);
    dims=3,
)
coil_gain = vec(
    sum(conj.(nominal_signal) .* measured_data; dims=1) ./
    sum(abs2, nominal_signal; dims=1),
)

forward(density) = dropdims(
    sum(basis .* reshape(density, 1, 1, :); dims=3);
    dims=3,
) .* permutedims(coil_gain)

# Finite-difference gradient descent
objective(parameters) =
    sum(abs2, forward(exp.(parameters)) - measured_data) /
    (2sum(abs2, measured_data))

function finite_difference_gradient(parameters)
    relative_step = cbrt(eps(eltype(parameters)))
    map(eachindex(parameters)) do index
        step = relative_step * max(abs(parameters[index]), one(eltype(parameters)))
        positive = copy(parameters)
        negative = copy(parameters)
        positive[index] += step
        negative[index] -= step
        (objective(positive) - objective(negative)) / (2step)
    end
end

parameters = log.(fill(sum(density_levels) / length(density_levels), length(density_levels)))
for iteration in 1:20
    loss = objective(parameters)
    local gradient = finite_difference_gradient(parameters)
    gradient_norm = norm(gradient)
    gradient_norm <= 1e-8 && break

    step = inv(gradient_norm)
    while objective(parameters .- step .* gradient) > loss
        step /= 2
    end
    parameters .-= step .* gradient
    println("Iteration $iteration: loss = $(objective(parameters)), gradient = $gradient_norm")
end

estimated_density = exp.(parameters)
reconstructed = copy(phantom)
for (original, estimated) in zip(density_levels, estimated_density)
    reconstructed.ρ[phantom.ρ .== original] .= estimated
end

output_file = joinpath(@__DIR__, "reconstructed_density.png")
savefig(
    plot_phantom_map(
        reconstructed,
        :ρ;
        view_2d=true,
        title="Finite-difference density reconstruction",
    ),
    output_file,
)

println("Estimated density levels: ", estimated_density)
println("Final loss: ", objective(parameters))
println("Saved: ", output_file)
