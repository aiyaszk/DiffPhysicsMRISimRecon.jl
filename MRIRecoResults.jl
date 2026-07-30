# MRIReco comparison of measured and node-simulated acquisitions

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using KomaMRI, MRICoilSensitivities
using Statistics: quantile

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

function select_profiles!(raw, lines)
    raw.profiles = raw.profiles[4:(3 + length(lines))]
    for (profile, line) in zip(raw.profiles, lines)
        profile.head.idx.kspace_encode_step_1 = UInt16(line)
    end
    nothing
end

magnitude_image(image) = abs.(Array(image[:, :, 1, 1, 1, 1]))

recon_size = (128, 128)
seq = read_seq(accelerated_2x_seq_file)
shots = Int(seq.DEF["EpiShots"])
acquired_shots = parse.(Int, split(seq.DEF["EpiAcquiredShots"], ',')) .- 1
acquired_lines = [
    line for shot in acquired_shots for line in shot:shots:(recon_size[2] - 1)
]

raw_reference = RawAcquisitionData(ISMRMRDFile(fully_sampled_mrd_file))
raw_reference.profiles = raw_reference.profiles[4:(3 + recon_size[2])]
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

raw_measured = RawAcquisitionData(ISMRMRDFile(accelerated_2x_mrd_file))
raw_simulated = RawAcquisitionData(ISMRMRDFile(simulated_mrd_file))
select_profiles!(raw_measured, acquired_lines)
select_profiles!(raw_simulated, acquired_lines)

acq_measured = AcquisitionData(raw_measured)
acq_simulated = AcquisitionData(raw_simulated)
acq_measured.traj[1].circular = false
acq_simulated.traj[1].circular = false

direct_params = Dict{Symbol,Any}(
    :reco => "direct",
    :reconSize => recon_size,
)
sense_params = Dict{Symbol,Any}(
    :reco => "multiCoil",
    :reconSize => recon_size,
    :senseMaps => sensitivity_maps,
    :iterations => 20,
    :densityWeighting => false,
    :toeplitz => false,
)

measured_direct = magnitude_image(reconstruction(acq_measured, direct_params))
measured_sense = magnitude_image(reconstruction(acq_measured, sense_params))
simulated_direct = magnitude_image(reconstruction(acq_simulated, direct_params))
simulated_sense = magnitude_image(reconstruction(acq_simulated, sense_params))

results = (
    ("acquisition_direct.png", reverse(measured_direct; dims=(1, 2)), "Measured direct (R=2)"),
    ("acquisition_sense.png", reverse(measured_sense; dims=(1, 2)), "Measured SENSE (R=2)"),
    ("simulated_mrd_direct.png", simulated_direct, "Node-simulated direct (R=2)"),
    ("simulated_mrd_sense.png", simulated_sense, "Node-simulated SENSE (R=2)"),
)

output_directory = joinpath(@__DIR__, "MRIRecoResults")
mkpath(output_directory)
foreach(results) do (filename, image, title)
    output_file = joinpath(output_directory, filename)
    savefig(
        plot_image(image; title, zmin=0, zmax=quantile(vec(image), 0.995)),
        output_file,
    )
    println("Saved: ", output_file)
end
