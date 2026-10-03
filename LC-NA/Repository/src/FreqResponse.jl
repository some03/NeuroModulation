module FreqResponse

export InputFreqSetting, step_freq_response

using Base: @kwdef
using Parameters: @unpack

@kwdef mutable struct InputFreqSetting
    """
    - `Fs::UInt32`: Sampling frequency (Hz).
    - `StopTime::Float32`: The total time duration for the simulation (seconds).
    - `Width::Float32`: The width of the step in the frequency response.
    - `f_diff::Float32`: The difference in frequency applied across the step.
    - `f_exc::Float32`: The base frequency before the step is applied.
    - `mid_point::Float32`: The midpoint in time where the step starts.
    - `f_inh::Float32`: The inhibitory frequency component applied consistently across the entire time period.
    """
    Fs::UInt32
    StopTime::Float32
    Width::Float32
    f_diff::Float32
    f_exc::Float32
    mid_point::Float32
    f_inh::Float32
end

function add_cosine(T::Float32, A::Float32, t::Number)::Number
    # -1をかけて位相を逆にする
    return -1*(A/2*cos(pi*t/T))+A/2
end

function step_freq_response(variable::InputFreqSetting)::Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}}
    """
    Generates a frequency response with a step function shape based on the given settings.
    
    # Arguments
    - `variable::InputFreqSetting`: An instance containing the configuration parameters for generating the frequency response.
        In detail, please check InputFreqSetting struct.
    
    # Returns
    - `Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}}`: A tuple containing:
        - `inputs`: A vector of the generated frequency values over time.
        - `currents`: A vector of the sinusoidal currents corresponding to the generated frequencies.
        - `noises`: A vector of inhibitory noise values (as integers) applied at each time step.
    
    # Note
    - The function uses a time step `dt` derived from the sampling frequency `Fs`. It iterates through the time range from `0` to `StopTime`, generating frequency values that follow a step function shape. The output `inputs` are the frequency values, `currents` are the sinusoidal responses corresponding to these frequencies, and `noises` are constant inhibitory values.
    """

    @unpack Fs,StopTime,Width,f_diff,f_exc,mid_point,f_inh = variable
    dt = 1/Fs;
    inputs = []
    currents = []
    noises = []
    @time for i = 0:dt:StopTime
        starts = mid_point-(Width/2);
        ends = mid_point+(Width/2);
        output = f_exc + add_cosine(Width,f_diff,i-starts) * ((i >= starts) & (i <= ends)) + f_diff * (i > ends);
        append!(inputs,output)
        append!(currents,sin(2*pi*output*i))
        append!(noises,f_inh)
    end
    return inputs,currents,noises
end

end