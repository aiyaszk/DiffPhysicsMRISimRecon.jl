using KomaMRIBase
using KomaMRICore
using Reactant

struct StaticCoilSens{A<:AbstractMatrix} <: AbstractRFReceiveSystem
    values::A
end

KomaMRIBase.get_n_coils(receiver::StaticCoilSens) = size(receiver.values, 2)
KomaMRIBase.get_sens(receiver::StaticCoilSens, _x, _y, _z) = receiver.values

function linear_stencil(axis, points)
    lower = Vector{Int}(undef, length(points))
    upper = similar(lower)
    weight = Vector{eltype(points)}(undef, length(points))
    origin = first(axis)
    spacing = step(axis)
    count = length(axis)
    for i in eachindex(points)
        coordinate = (points[i] - origin) / spacing + 1
        if coordinate <= 1
            lower[i] = upper[i] = 1
            weight[i] = 0
        elseif coordinate >= count
            lower[i] = upper[i] = count
            weight[i] = 0
        else
            lower[i] = floor(Int, coordinate)
            upper[i] = lower[i] + 1
            weight[i] = coordinate - lower[i]
        end
    end
    return lower, upper, weight
end

function bilinear_stencil(axis_x, axis_y, spin_x, spin_y)
    ix0, ix1, wx = linear_stencil(axis_x, spin_x)
    iy0, iy1, wy = linear_stencil(axis_y, spin_y)
    linear_index(ix, iy) = ix .+ (iy .- 1) .* length(axis_x)
    return (;
        i00=linear_index(ix0, iy0),
        i10=linear_index(ix1, iy0),
        i01=linear_index(ix0, iy1),
        i11=linear_index(ix1, iy1),
        wx,
        wy,
    )
end

function interpolate_voxels(x, stencil)
    wx = stencil.wx
    wy = stencil.wy
    return (
        (1 .- wx) .* (1 .- wy) .* x[stencil.i00] .+
        wx .* (1 .- wy) .* x[stencil.i10] .+
        (1 .- wx) .* wy .* x[stencil.i01] .+
        wx .* wy .* x[stencil.i11]
    )
end

struct ADSequence{Excitation,SampleCount,Gx,Gy,Gz,B1,Df,Psi,ADC,T,Dt}
    Gx::Gx
    Gy::Gy
    Gz::Gz
    B1::B1
    Δf::Df
    ψ::Psi
    ADC::ADC
    t::T
    Δt::Dt
end

is_excitation(::ADSequence{Excitation}) where {Excitation} = Excitation
sample_count(::ADSequence{Excitation,SampleCount}) where {Excitation,SampleCount} = SampleCount
simulation_complex(::Val{:f32}) = ComplexF32
simulation_complex(::Val{:f64}) = ComplexF64
simulation_complex(::Val{:bigfloat}) = Complex{BigFloat}

function device_sequence(seq::DiscreteSequence, excitation)
    device(values) = Reactant.to_rarray(values)
    values = (
        device(seq.Gx),
        device(seq.Gy),
        device(seq.Gz),
        device(seq.B1),
        device(seq.Δf),
        device(seq.ψ),
        device(Int32.(seq.ADC)),
        device(seq.t),
        device(seq.Δt),
    )
    sequence_type = Core.apply_type(
        ADSequence,
        excitation,
        count(seq.ADC[2:end]),
        map(typeof, values)...,
    )
    return sequence_type(values...)
end

function prepare_simulation(seq, obj, sys, sim_params, physio)
    settings = KomaMRICore.default_sim_params(copy(sim_params))
    seq = KomaMRIBase.resolve_triggers(seq, physio)
    sampling_rule = KomaMRICore.simulation_sampling_rule(settings["sim_method"], settings)
    seqd = KomaMRIBase.discretize(
        seq;
        sampling_rule,
        motion=obj.motion,
        freq_in_phase=settings["freq_in_phase"],
    )
    parts, excitation = KomaMRICore.get_sim_ranges(
        seqd;
        max_block_length=settings["max_block_length"],
        max_rf_block_length=settings["max_rf_block_length"],
        eval_intervals_per_step=KomaMRICore.eval_intervals_per_step(settings["sim_method"]),
    )
    transform = KomaMRICore.simulation_precision_transform(Val(Symbol(settings["precision"])))
    host_blocks = Tuple(transform(seqd[p]) for p in parts)
    samples = Tuple(count(block.ADC[2:end]) for block in host_blocks)
    blocks = Tuple(device_sequence(block, excite) for (block, excite) in zip(host_blocks, excitation))
    return (;
        blocks,
        samples,
        total_samples=sum(samples),
        max_block_length=maximum(length.(parts)),
        max_adc_samples=maximum(samples; init=0),
        precession_groupsize=settings["gpu_groupsize_precession"],
        excitation_groupsize=settings["gpu_groupsize_excitation"],
        precision=Symbol(settings["precision"]),
        ncoils=get_n_coils(sys.receiver),
    )
end

function simulate_prepared(obj, sys, prepared)
    transform = KomaMRICore.simulation_precision_transform(Val(prepared.precision))
    obj = transform(obj)
    method = KomaMRICore.Bloch()
    backend = KomaMRICore.KA.CPU()
    M, obj = KomaMRICore.initialize_spins_state(obj, method)
    signal = similar(
        obj.ρ,
        simulation_complex(Val(prepared.precision)),
        prepared.total_samples,
        prepared.ncoils,
    )
    fill!(signal, zero(eltype(signal)))
    prealloc = KomaMRICore.prealloc(
        method,
        backend,
        obj,
        M,
        prepared.max_block_length,
        prepared.max_adc_samples,
        min(prepared.precession_groupsize, prepared.excitation_groupsize),
        sys,
    )
    sample = 1
    for block in eachindex(prepared.blocks)
        count = prepared.samples[block]
        block_signal = similar(signal, count, prepared.ncoils)
        fill!(block_signal, zero(eltype(block_signal)))
        if is_excitation(prepared.blocks[block])
            Reactant.@trace KomaMRICore.run_spin_excitation!(
                obj,
                prepared.blocks[block],
                block_signal,
                M,
                sys,
                method,
                prepared.excitation_groupsize,
                backend,
                prealloc,
            )
        else
            Reactant.@trace KomaMRICore.run_spin_precession!(
                obj,
                prepared.blocks[block],
                block_signal,
                M,
                sys,
                method,
                prepared.precession_groupsize,
                backend,
                prealloc,
            )
        end
        if count > 0
            signal[sample:(sample + count - 1), :] .= block_signal
            sample += count
        end
    end
    return signal
end

function traced_acquire!(sig, sample, Mxy, receiver::StaticCoilSens, acquire)
    signal = vec(transpose(Mxy) * receiver.values)
    for coil in axes(sig, 2)
        current = Reactant.@allowscalar sig[sample, coil]
        Reactant.@allowscalar sig[sample, coil] = ifelse(acquire, signal[coil], current)
    end
    return nothing
end

function KomaMRICore.run_spin_precession!(
    p::Phantom,
    seq::ADSequence,
    sig::AbstractArray{<:Reactant.TracedRNumber},
    M::KomaMRICore.Mag,
    sys,
    ::KomaMRICore.Bloch,
    _groupsize,
    ::KomaMRICore.KA.CPU,
    prealloc::KomaMRICore.PreallocResult,
)
    T = eltype(p.ρ)
    Bz_old = prealloc.Bz_old
    Bz_new = prealloc.Bz_new
    ϕ = prealloc.ϕ
    Mxy = prealloc.M.xy
    ΔBz = prealloc.ΔBz
    receiver = sys.receiver
    phase_scale = T(-Float64(π) * γ)
    fill!(ϕ, zero(T))
    block_time = zero(T)
    sample = similar(seq.ADC, Int64, 1)
    fill!(sample, 0)
    t_start = Reactant.@allowscalar seq.t[1]
    Gx_start = Reactant.@allowscalar seq.Gx[1]
    Gy_start = Reactant.@allowscalar seq.Gy[1]
    Gz_start = Reactant.@allowscalar seq.Gz[1]
    x, y, z = KomaMRICore.spin_coordinates(p.motion, p.x, p.y, p.z, t_start)
    @. Bz_old = x * Gx_start + y * Gy_start + z * Gz_start + ΔBz
    Reactant.@allowscalar Reactant.@trace checkpointing=true for i in eachindex(seq.Δt)
        x, y, z = KomaMRICore.spin_coordinates(p.motion, p.x, p.y, p.z, seq.t[i + 1])
        @. Bz_new = x * seq.Gx[i + 1] + y * seq.Gy[i + 1] + z * seq.Gz[i + 1] + ΔBz
        @. ϕ += (Bz_old + Bz_new) * phase_scale * seq.Δt[i]
        block_time += seq.Δt[i]
        if !isempty(sig)
            acquire = !iszero(seq.ADC[i + 1])
            sample_value = Reactant.@allowscalar sample[1]
            sample_value += ifelse(acquire, 1, 0)
            Reactant.@allowscalar sample[1] = sample_value
            @. Mxy = exp(-block_time / p.T2) * M.xy * cis(ϕ)
            KomaMRICore.outflow_spin_reset!(Mxy, seq.t[i + 1], p.motion)
            traced_acquire!(sig, max(sample_value, 1), Mxy, receiver, acquire)
        end
        Bz_old .= Bz_new
    end
    @. M.xy = M.xy * exp(-block_time / p.T2) * cis(ϕ)
    @. M.z = M.z * exp(-block_time / p.T1) + p.ρ * (T(1) - exp(-block_time / p.T1))
    KomaMRICore.outflow_spin_reset!(M, seq.t', p.motion; replace_by=p.ρ)
    return nothing
end

function KomaMRICore.run_spin_excitation!(
    p::Phantom,
    seq::ADSequence,
    sig::AbstractArray{<:Reactant.TracedRNumber},
    M::KomaMRICore.Mag,
    sys,
    ::KomaMRICore.Bloch,
    _groupsize,
    ::KomaMRICore.KA.CPU,
    prealloc::KomaMRICore.BlochCPUPrealloc,
)
    T = eltype(p.ρ)
    Bz = prealloc.Bz_old
    B = prealloc.Bz_new
    φ_half = prealloc.ϕ
    α = prealloc.Rot.α
    β = prealloc.Rot.β
    ΔBz = prealloc.ΔBz
    Maux_xy = prealloc.M.xy
    Maux_z = prealloc.M.z
    receiver = sys.receiver
    πT = T(Float64(π))
    phase_scale = T(-Float64(π) * γ)
    sample = similar(seq.ADC, Int64, 1)
    fill!(sample, 0)
    ψ_start = Reactant.@allowscalar seq.ψ[1]
    @. M.xy = M.xy * cis(-ψ_start)
    Reactant.@allowscalar for i in eachindex(seq.Δt)
        x, y, z = KomaMRICore.spin_coordinates(p.motion, p.x, p.y, p.z, seq.t[i])
        B1 = Reactant.@allowscalar seq.B1[i]
        @. Bz = (seq.Gx[i] * x + seq.Gy[i] * y + seq.Gz[i] * z) + ΔBz - seq.Δf[i] / T(γ)
        @. B = sqrt(abs2(B1) + Bz^2)
        @. φ_half = phase_scale * (B * seq.Δt[i])
        @. α = cos(φ_half)
        @. B = phase_scale * seq.Δt[i] * sinc(φ_half / πT)
        @. α -= complex(zero(Bz), Bz * B)
        @. β = complex(imag(B1) * B, -real(B1) * B)
        KomaMRICore.mul!(KomaMRICore.Spinor(α, β), M, Maux_xy, Maux_z)
        @. M.xy = M.xy * exp(-seq.Δt[i] / p.T2)
        @. M.z = M.z * exp(-seq.Δt[i] / p.T1) + p.ρ * (T(1) - exp(-seq.Δt[i] / p.T1))
        KomaMRICore.outflow_spin_reset_at!(M, seq.t, i + 1, p.motion; replace_by=p.ρ)
        if !isempty(sig)
            acquire = !iszero(seq.ADC[i + 1])
            sample_value = Reactant.@allowscalar sample[1]
            sample_value += ifelse(acquire, 1, 0)
            Reactant.@allowscalar sample[1] = sample_value
            traced_acquire!(sig, max(sample_value, 1), M.xy, receiver, acquire)
        end
    end
    ψ_end = Reactant.@allowscalar seq.ψ[end]
    @. M.xy = M.xy * cis(ψ_end)
    return nothing
end
