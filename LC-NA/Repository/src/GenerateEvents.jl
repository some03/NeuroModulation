module GenerateEvents

export generate_events_tonic, generate_events_phasic

using DifferentialEquations


function generate_events_tonic(
    freq_inputs::Vector{Float32},
    freq_noises::Vector{Float32},
    dt_signal::Float64,
    neuron_count::Int64
)::Tuple{Array{Int64, 2}, Array{Int64, 2}}
    """Generate input and tonic noise spike responses for each neuron.
    # Arguments
    - `freq_inputs::Vector{Float32}`: Input frequency response.
    - `freq_noises::Vector{Float32}`: Noise response to be added to the input.
    - `dt_signal::Number`: The time step for the signal processing.
        Attention: dt set to "msec", it must be converted into sec for extracting firing index.
                   Therefore, for example, if dt = 0.05ms, dt_signal = 0.05 * 1e-3.
    - `neuron_count::Int`: The number of neuron for the first layer.
        Default to Ne+Ni=100.
    
    # Returns
    - `Tuple{Array{Int64,2}, Vector{Int64}}`: A tuple containing:
        - `spikes`: 
            A 2D array where each row corresponds to a time step.
            Each column corresponds to a neuron (Ne + Ni), indicating whether a spike occurred (1) or not (0).
        - `noise_spikes`: A vector indicating the noise spike.
    
    # Note
    - The first index of `noise_spikes` corresponds to the noise spikes because the same noise effect is projected into all neurons equally.
    """
    # Note: example: dt_signal=dt*1e-3の字数がそのままdigitsの数に対応するので注意
    ####################################
    # Generate input and noise signal
    ####################################
    output_spikes = isone.(zeros(length(freq_inputs), neuron_count));
    noise_spikes = isone.(zeros(length(freq_inputs), neuron_count));
    # Generate event time for each neuron.
    for i in 1:neuron_count
        L = length(freq_inputs);
        input_event_times, _ = pprocess_inhomopoisson(freq_inputs, dt_signal,L);
        noise_event_times, _ = pprocess_inhomopoisson(freq_noises, dt_signal,L);
        for r in input_event_times
            ind = round(r/dt_signal);
            ind = convert(Int64, ind);
            output_spikes[ind+1, i] = 1;  # +1しないと0にアクセスすることになるので範囲外になるので注意
        end
        for r in noise_event_times
            # dt*1e-3で1e-5なので、digitsは5にする必要があるので注意！
            ind_noise = round(r/dt_signal);
            ind_noise = convert(Int64, ind_noise);
            noise_spikes[ind_noise+1, i] = 1;
        end
    end
    # return output_spikes, noise_spikes[:,1]
    return output_spikes, noise_spikes
end

function generate_events_phasic(
    freq_inputs::Vector{Float32},
    dt_signal::Float64,
    neuron_count::Int64,
    f_inh::Float64,
    T::Float64,
    dt::Float64,
)::Tuple{Array{Int64, 2}, Array{Int64, 2}}
    """Generate input and phasic noise spike responses for each neuron.
    # Arguments
    - `freq_inputs::Vector{Float32}`: Input frequency response.
    - `dt_signal::Number`: The time step for the signal processing.
        Attention: dt set to "msec", it must be converted into sec for extracting firing index.
                   Therefore, for example, if dt = 0.05ms, dt_signal = 0.05 * 1e-3.
    - `neuron_count::Int`: The number of neuron for the first layer.
        Default to Ne+Ni=100.
    - `f_inh::Float64`: The noise frequency.
        Default to 100.0.
    - `T::Float64`: The overall time span.
        Default to 1600.0.
    - `dt::Float64`: The time step.
        Default to dt=0.05[msec]
    
    # Returns
    - `Tuple{Array{Int64,2}, Vector{Int64}}`: A tuple containing:
        - `spikes`: 
            A 2D array where each row corresponds to a time step.
            Each column corresponds to a neuron (Ne + Ni), indicating whether a spike occurred (1) or not (0).
        - `noise_spikes`: A vector indicating the noise spike.
    
    # Note
    - The first index of `noise_spikes` corresponds to the noise spikes because the same noise effect is projected into all neurons equally.
    """
    # Note: example: dt_signal=dt*1e-3の字数がそのままdigitsの数に対応するので注意
    ####################################
    # Generate input and noise signal
    ####################################
    output_spikes = isone.(zeros(length(freq_inputs), neuron_count))
    output_noise = isone.(zeros(length(freq_inputs), neuron_count))
    for i in 1:(neuron_count)
        μ = f_inh
        σ = f_inh
        Θ = 0.1
        W = OrnsteinUhlenbeckProcess(Θ,μ,σ,0.0,f_inh)
        prob = NoiseProblem(W,(0.0,T))
        sol = solve(prob;dt=dt)

        L = length(freq_inputs);
        res, _ = pprocess_inhomopoisson(freq_inputs, dt_signal,L);
        # Convert Vector{Float64} into Vector{Float32}
        res_noise, _ = pprocess_inhomopoisson(Float32.(sol.u[2:end]),dt_signal,L);
        for r in res
            ind = round(r/dt_signal);
            ind = convert(Int64, ind);
            output_spikes[ind+1, i] = 1;
        end
        for r in res_noise
            ind2 = round(r/dt_signal); # dt*1e-3で1e-5なので、digitsは5にする必要があるので注意！
            ind2 = convert(Int64, ind2);
            output_noise[ind2+1, i] = 1;
        end
    end
    # return output_spikes, output_noise[:,1]
    return output_spikes, output_noise
end


# ベクトルは、Juliaにおける特殊な Array（1次元の配列）であり、Array{T, 1} としても表現される
function pprocess_inhomopoisson(
    lambda::Vector{Float32},
    dt::Number,
    L::Number
)::Tuple{Vector{Float32}, Vector{Float32}}
    """Generates firing times from inhomogeneous Poisson process.

    # Arguments
    - `lambda::Vector{Float32}`: Firing Rate for time-series (Hz).
    - `dt::Number`: The time step.
    - `L::Number`: The number of time step.
    
    # Returns
    - `Tuple{Vector{Float32}, Vector{Float32}}`: A tuple containing:
        - event_times: Event time indices.
        - rs: Firing rates.
    """
    # ここでrは各時刻の発火頻度\lambda(t)λ(t)、
    # Rはその積分値を意味する。 以下にその生成結果を示す
    event_times = [0.0];
    rs = zeros(L);
    R = 0;
    rs[1] = lambda[1];

    eta = - log(rand());
    for i = 2:L-1
        tau = i*dt - event_times[end];

        r = lambda[i];
        R = R + r*dt;

        if eta <= R
            push!(event_times,i*dt);

            R = 0.0;
            eta = - log(rand());
        end
        rs[i] = r;
    end

    return event_times, rs
end

end