push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using HodgkinHuxley, DelayConnection

using Distributions, Random
using Plots

dt, T = 0.05, 100.0 # タイムステップ, シミュレーション時間 (msec)
nt = UInt32(T/dt) # number of timesteps
N = UInt32(5) # ニューロンの数

# 入力刺激
t = (1:nt)*dt
Ie = repeat(10f0 * ((t .> 40) - (t .> 80)), 1, N)  # injection current


# modelの定義
neurons1 = HodgkinHuxley.HH{Float32}(N=N)
neurons2 = HodgkinHuxley.HH{Float32}(N=N)

# delay set to 2 msec.
delay_exc_connection = DelayConnection.Delay_Connection(N, 2.0, T, dt);

v_arr1 = zeros(Float64, nt, N)
v_arr2 = zeros(Float64, nt, N)

i=1;
@time for i in 1:nt
    HodgkinHuxley.rk_update!(neurons1, neurons1.param, Ie[i, :], dt)
    v_arr1[i, :] = neurons1.v
    if i > 1
        fire = (v_arr1[i-1, :] .< -40) .& (v_arr1[i, :] .> -40);
        ret = DelayConnection.Delay_Connection_call!(delay_exc_connection, collect(fire), UInt32(i))
        # Memo: ここのパラメータについて理解しておく
        HodgkinHuxley.rk_update!(neurons2, neurons2.param, 10/dt*ret, dt)
        v_arr2[i, :] = neurons2.v
    end
end

t = Array{Float32}(1:nt)*dt
plot(t, v_arr1[:, 1], color="red", label="neuron1")
plot!(t, v_arr2[:, 1], color="blue", label="neuron2", legend=true)

savefig("delay_sample.png")