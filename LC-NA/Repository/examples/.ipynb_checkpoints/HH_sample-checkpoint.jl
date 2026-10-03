push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using HodgkinHuxley

using Plots

T = 600 # ms
dt = 0.05 # ms
nt = Int(T/dt) # number of timesteps
N = 10 # ニューロンの数

# 入力刺激
t = (1:nt)*dt
Ie = repeat(10f0 * ((t .> 50) - (t .> 200)) + 35f0 * ((t .> 250) - (t .> 400)), 1, N)  # injection current

# 記録用
varr, gatearr = zeros(nt, N), zeros(nt, 3, N)

# modelの定義
neurons = HodgkinHuxley.HH{Float32}(N=N)
# simulation
@time for i = 1:nt
    HodgkinHuxley.rk_update!(neurons, neurons.param, Ie[i, :], dt)
    varr[i, :] = neurons.v
    gatearr[i, :, :] = [neurons.m'; neurons.h'; neurons.n'] # 修正
end

p1 = plot(t, varr[:, 1], color="black")
labellist=["orange" "blue" "green"]
p2 = plot(t, gatearr[:, 1, 1], color=labellist[1])
for i in 2:3
    p2 = plot!(t, gatearr[:, i, 1], color=labellist[i])
end; 
p3 = plot(t, Ie[:, 1], color="green")
plot(p1, p2, p3,
    xlabel = ["" "" "Times (ms)"], 
    ylabel= ["Membrane\n potential (mV)" "Gating\n value" "Injection\n current (pA)"],
    layout = grid(3, 1, heights=[0.4, 0.4, 0.2]), guidefont=font(6), legend=false, size=(500,300))
savefig("HH_sample.png")
