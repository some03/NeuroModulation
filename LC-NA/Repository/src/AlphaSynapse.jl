module AlphaSynapse

using Base: @kwdef
using Parameters: @unpack

@kwdef struct Synapse_Parameter{FT}
    t_exc::FT = 2.0*1e-3  # sec
    t_inh::FT = 5.0*1e-3 # sec
end

@kwdef mutable struct Synapse{FT}
    param::Synapse_Parameter = Synapse_Parameter{FT}()
    Ne::UInt32 # Excニューロンの数
    Ni::UInt32 # Inhニューロンの数
    T::Float32 # Time interval
    dt::Float32 # dt
    nt::UInt32 = round(Int64,T/dt) #time count
    g_syn::Matrix{FT} = zeros(Float32, Ne+Ni, nt)
    hr::Matrix{FT} = zeros(Float32, Ne+Ni, nt)
end 

function Synaptic_update!(variable::Synapse, param::Synapse_Parameter, spikes, t)
    @unpack Ne, Ni, T, dt, nt, g_syn, hr = variable
    @unpack t_exc, t_inh = param
    
    if Ne>0
        @inbounds @simd for i = 1:Ne
            ########### シナプス動態の導出 ###########
            hr[i,t+1] = (1 - (dt/t_exc)) * hr[i,t] + (1/(t_exc)^2) * spikes[i]
            g_syn[i,t+1] = (1 - (dt/t_exc)) * (g_syn[i,t]) + hr[i,t] * dt
            ########################################
        end
    end

    if Ni>0
        @inbounds @simd for i = Ne+1:Ne+Ni
            ########### シナプス動態の導出 ###########
            hr[i,t+1] = (1 - (dt/t_inh)) * hr[i,t] + (1/(t_inh)^2) * spikes[i]
            g_syn[i,t+1] = (1 - (dt/t_inh)) * (g_syn[i,t]) + hr[i,t] * dt
            ########################################
        end
    end
end

end
