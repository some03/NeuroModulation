push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using AlphaSynapse

using Distributions, Random
using Plots

dt, T = 0.05*1e-3, 100*1e-3 # タイムステップ, シミュレーション時間 (sec)
nt = Int(T/dt) # シミュレーションの総ステップ

synapses_exc = AlphaSynapse.Synapse{Float64}(Ne=1, Ni=0, T=T, dt=dt)


V_exc = 0
V_inh = -70
Er = -65

d = Normal(1.0, 0.5)

params_g = rand(d)

r_double= zeros(nt)
for t in 1:nt-1
    spike = ifelse(t == 1, 1, 0)
    AlphaSynapse.Synaptic_update!(synapses_exc, synapses_exc.param, collect(spike)[:], t)
    r_double[t+1] = (synapses_exc.g_syn[t+1] .* (V_exc-Er) .* (params_g ./ (4*pi*((20)^2))))
    # r_double[t+1] = (synapses_exc.g_syn[t+1] .* (V_exc-Er) .* params_g)
    #synapses_exc.hr[t] = synapses_exc.hr[t]
    #r_double[t+1] = r_double[t]*exp(-dt/tr) + hr[t]*dt
    #hr[t+1] = hr[t]*exp(-dt/td) + spike/(tr*td)
end 

p1 = plot((1:nt)*dt, r_double, color="red",label="alpha function")
plot(p1 , guidefont=font(8),
xlabel = "Time [sec]", 
ylabel = "synaptic dynamics",legend=true, size=(300,300))

savefig("synapse_sample.png")
