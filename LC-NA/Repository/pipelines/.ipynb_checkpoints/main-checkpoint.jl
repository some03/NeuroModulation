push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using ArgParse
using Base: @kwdef
using Parameters: @unpack
using Plots
using LinearAlgebra
using Distributions, Random
using Dates
using JSON
using Statistics
using NPZ
using DifferentialEquations
using Serialization

using Base.Threads
Threads.nthreads()

using AlphaSynapse
using DelayConnection
using FreqResponse
using GenerateEvents
using HodgkinHuxley


function intra_layer_update!(
    varr, noises, spikes, efficacy, neurons, synapses_gen, synapses_noise,
    synapses_exc, synapses_inh, t, input_weight, feedback_weight, feedback_weight_inh, 
    V_exc, Er, V_inh, dt, threshold, delay_exc_connection, delay_inh_connection, delta
)
    # ここのシナプスモデルの出力は、mS/cm^2なのかどうか
    AlphaSynapse.Synaptic_update!(synapses_gen, synapses_gen.param, spikes, t)
    AlphaSynapse.Synaptic_update!(synapses_noise, synapses_gen.param, noises, t)

    # gsyn_exc = input_weight * efficacy .* (synapses_gen.g_syn[:,t] .+ synapses_noise.g_syn[:,t])
    gsyn_exc = input_weight * efficacy .* (synapses_gen.g_syn[:,t])
    gsyn_exc .+= feedback_weight * efficacy .* synapses_exc.g_syn[:,t]
    gsyn_inh = feedback_weight_inh * synapses_inh.g_syn[:,t]
    # Ieは数μA/cm^2から数十μA/cm^2になることが望ましい
    Ie = ((gsyn_exc .* (V_exc-Er)) .+ (gsyn_inh .* (V_inh-Er)))
    HodgkinHuxley.rk_update!(neurons, neurons.param, Ie, dt)
    
    if t > 1
        post_fire = (varr[t-1, :] .< threshold) .& (neurons.v .> threshold)
        #input_spikes1 = DelayConnection.Delay_Connection_call!(delay_exc_connection, collect(spikes.+noises .> 0)[:], UInt32(t))
        #input_spikes2 = DelayConnection.Delay_Inh_Connection_call!(delay_inh_connection, collect(spikes.+noises .> 0)[:], UInt32(t))
        input_spikes1 = DelayConnection.Delay_Connection_call!(delay_exc_connection, collect(post_fire.+noises .> 0)[:], UInt32(t))
        input_spikes2 = DelayConnection.Delay_Inh_Connection_call!(delay_inh_connection, collect(post_fire.+noises .> 0)[:], UInt32(t))
        AlphaSynapse.Synaptic_update!(synapses_exc, synapses_exc.param, input_spikes1, t)
        AlphaSynapse.Synaptic_update!(synapses_inh, synapses_inh.param, input_spikes2, t)
    
        for j in 1:length(efficacy)
            if post_fire[j] > 0
                efficacy[j] = efficacy[j] * delta
            else
                efficacy[j] = minimum([efficacy[j] + 1e-4, 1])
            end
        end
    end
    
    return neurons.v, efficacy, Ie
end


function intra_layer(input_length, input_fire, noise_fire, params, delta)
    # Define between layer
    delay_exc_connection = DelayConnection.Delay_Connection(
        params.Ne+params.Ni,
        2.0,
        params.T,
        params.dt
    )
    delay_inh_connection = DelayConnection.Delay_Inh_Connection(
        params.Ne+params.Ni,
        5.0,
        params.T,
        params.dt
    )
    neurons = HodgkinHuxley.HH{Float32}(N=params.Ne+params.Ni)
    synapses_gen = AlphaSynapse.Synapse{Float32}(Ne=params.Ne+params.Ni, Ni=0, T=params.T*1e-3, dt=params.dt*1e-3)
    synapses_noise = AlphaSynapse.Synapse{Float32}(Ne=params.Ne+params.Ni, Ni=0, T=params.T*1e-3, dt=params.dt*1e-3)
    synapses_exc = AlphaSynapse.Synapse{Float32}(Ne=params.Ne+params.Ni, Ni=0, T=params.T*1e-3, dt=params.dt*1e-3)
    synapses_inh = AlphaSynapse.Synapse{Float32}(Ne=0, Ni=params.Ne+params.Ni, T=params.T*1e-3, dt=params.dt*1e-3)
    
    varr = zeros(Float32, input_length, params.Ne+params.Ni)
    uarr = zeros(Float32, input_length, params.Ne+params.Ni)
    connect_exc = zeros(Float32, params.Ne+params.Ni, params.Ne+params.Ni)
    connect_inh = zeros(Float32, params.Ne+params.Ni, params.Ne+params.Ni)
    connect_input = zeros(Float32, params.Ne+params.Ni, params.Ne+params.Ni)

    for i in 1:params.Ne+params.Ni
        inh_list = rand(1:params.Ne+params.Ni, params.NiC)
        for ind in inh_list
            connect_inh[i,ind] += 1
        end
        exc_list = rand(setdiff(Set(collect(1:params.Ne+params.Ni)), Set(inh_list)), params.NeC)
        for ind in exc_list
            connect_exc[i,ind] += 1
        end
    end
    
    for i in 1:params.Ne+params.Ni
        input_list = rand(1:params.Ne+params.Ni, params.NiE)
        for ind in input_list
            connect_input[i,ind] += 1
        end
    end

    efficacy = ones(Float64, params.Ne+params.Ni)
    g_rand = rand(params.d, params.Ne+params.Ni, params.Ne+params.Ni)
    g_input = rand(params.d, params.Ne+params.Ni, params.Ne+params.Ni)
    g_noise = rand(params.d, params.Ne+params.Ni, params.Ne+params.Ni)
    surface = 4*pi*(20^2) * (params.Ne+params.Ni)
    # surface = 4*pi*(20^2)
    input_weight = g_input ./ surface
    noise_weight = g_noise ./ surface
    feedback_weight = connect_exc .* g_rand ./ surface
    feedback_weight_inh = connect_inh .* 5 ./ surface
    
    # main loop
    @inbounds for i in 1:params.nt-1
        i = Int(i)
        varr[i, :], efficacy, uarr[i, :] = intra_layer_update!(
            varr, noise_fire[i,:], input_fire[i,:], efficacy, neurons, 
            synapses_gen, synapses_noise, synapses_exc, synapses_inh, i,
            input_weight, feedback_weight, feedback_weight_inh, params.V_exc, params.Er, params.V_inh, params.dt, params.threshold,
            delay_exc_connection, delay_inh_connection, delta
        )
    end
    
    # Return firing
    # 前後で発火を判断してるので、得られる長さは最大長-1になることに注意
    fire = zeros(Float32, input_length, params.Ne+params.Ni)
    fire = (varr[1:Int(params.nt)-1, :] .< params.threshold) .& (varr[2:Int(params.nt), :] .> params.threshold)

    return Bool.(fire)
end

function between_layer_update!(
    varr, noises, spikes, efficacy, neurons, 
    synapses_gen, synapses_noise, synapses_exc, synapses_inh, t, input_weight, feedback_weight, feedback_weight_inh,
    V_exc, Er, V_inh, dt, threshold, delay_exc_connection, delay_inh_connection, delta
)
    AlphaSynapse.Synaptic_update!(synapses_gen, synapses_gen.param, spikes, t)
    AlphaSynapse.Synaptic_update!(synapses_noise, synapses_gen.param, noises, t)

    # gsyn_exc = input_weight * efficacy .* (synapses_gen.g_syn[:,t] .+ synapses_noise.g_syn[:,t])
    gsyn_exc = input_weight * efficacy .* (synapses_gen.g_syn[:,t])
    gsyn_exc .+= feedback_weight * efficacy .* synapses_exc.g_syn[:,t]
    gsyn_inh = feedback_weight_inh * synapses_inh.g_syn[:,t]

    Ie = ((gsyn_exc .* (V_exc-Er)) .+ (gsyn_inh .* (V_inh-Er)))
    HodgkinHuxley.rk_update!(neurons, neurons.param, Ie, dt)

    if t > 1
        post_fire = (varr[t-1, :] .< threshold) .& (neurons.v .> threshold)
        #input_spikes1 = DelayConnection.Delay_Connection_call!(delay_exc_connection, collect(spikes.+noises .> 0)[:], UInt32(t))
        #input_spikes2 = DelayConnection.Delay_Inh_Connection_call!(delay_inh_connection, collect(spikes.+noises .> 0)[:], UInt32(t))
        input_spikes1 = DelayConnection.Delay_Connection_call!(delay_exc_connection, collect(post_fire.+noises .> 0)[:], UInt32(t))
        input_spikes2 = DelayConnection.Delay_Inh_Connection_call!(delay_inh_connection, collect(post_fire.+noises .> 0)[:], UInt32(t))
        AlphaSynapse.Synaptic_update!(synapses_exc, synapses_exc.param, input_spikes1, t)
        AlphaSynapse.Synaptic_update!(synapses_inh, synapses_inh.param, input_spikes2, t)

        for j in 1:length(efficacy)
            if post_fire[j] > 0
                efficacy[j] = efficacy[j] * delta
            else
                efficacy[j] = minimum([efficacy[j] + 1e-4, 1])
            end
        end
    end

    return neurons.v, efficacy, Ie
end

function between_layer(input_length, input_fire, noise_fire, params, delta)
    # Initialize layer parameters
    delay_exc_connection = DelayConnection.Delay_Connection(
        UInt32(params.Ne+params.Ni),
        2.0,
        params.T,
        params.dt
    )
    delay_inh_connection = DelayConnection.Delay_Inh_Connection(
        UInt32(params.Ne+params.Ni),
        5.0,
        params.T,
        params.dt
    )
    neurons = HodgkinHuxley.HH{Float32}(N=params.Ne+params.Ni)
    synapses_gen = AlphaSynapse.Synapse{Float64}(Ne=params.Ne+params.Ni, Ni=0, T=params.T*1e-3, dt=params.dt*1e-3)
    synapses_noise = AlphaSynapse.Synapse{Float64}(Ne=params.Ne+params.Ni, Ni=0, T=params.T*1e-3, dt=params.dt*1e-3)
    synapses_exc = AlphaSynapse.Synapse{Float64}(Ne=params.Ne+params.Ni, Ni=0, T=params.T*1e-3, dt=params.dt*1e-3)
    synapses_inh = AlphaSynapse.Synapse{Float64}(Ne=0, Ni=params.Ne+params.Ni, T=params.T*1e-3, dt=params.dt*1e-3)
    
    varr = zeros(Float32, input_length, params.Ne+params.Ni)
    uarr = zeros(Float32, input_length, params.Ne+params.Ni)
    connect_exc = zeros(Float32, params.Ne+params.Ni, params.Ne+params.Ni)
    connect_inh = zeros(Float32, params.Ne+params.Ni, params.Ne+params.Ni)
    connect_input = zeros(Float32, params.Ne+params.Ni, params.Ne+params.Ni)

    for i in 1:params.Ne+params.Ni
        inh_list = rand(1:params.Ne+params.Ni, params.NiC)
        for ind in inh_list
            connect_inh[i,ind] += 1
        end
        exc_list = rand(setdiff(Set(collect(1:params.Ne+params.Ni)),Set(inh_list)), params.NeC)
        for ind in exc_list
            connect_exc[i,ind] += 1
        end
    end

    efficacy = ones(Float64, params.Ne+params.Ni)
    g_rand = rand(params.d, params.Ne+params.Ni, params.Ne+params.Ni)
    g_input = rand(params.d, params.Ne+params.Ni, params.Ne+params.Ni)
    g_noise = rand(params.d, params.Ne+params.Ni, params.Ne+params.Ni)
    surface = 4*pi*(20^2) * (params.Ne+params.Ni)  # 全体の影響を評価してるので、今回設定してる100個の細胞だけ掛け算する
    # surface = 4*pi*(20^2)
    input_weight = g_input ./ surface
    noise_weight = g_noise ./ surface
    feedback_weight = connect_exc .* g_rand ./ surface
    feedback_weight_inh = connect_inh .* 5 ./ surface

    fire_propagation_rate = Categorical([0.3, 0.7])
    filter_connect = (rand(fire_propagation_rate, params.Ne+params.Ni, params.Ne+params.Ni).-1)

    # Main loop
    @inbounds for i = 1:params.nt-1
        filter_spike = filter_connect * input_fire[i,:]
        filter_spike = isone.(ifelse.(filter_spike .> 0, 1, 0))
        varr[i, :], efficacy, uarr[i, :] = between_layer_update!(
            varr, noise_fire[i,:], filter_spike, efficacy, neurons,
            synapses_gen, synapses_noise, synapses_exc, synapses_inh,
            i, input_weight, feedback_weight, feedback_weight_inh,
            params.V_exc, params.Er, params.V_inh, params.dt, params.threshold, delay_exc_connection, delay_inh_connection, delta
        )
    end

    # Return firing
    fire = (varr[1:Int(params.nt)-1, :] .< params.threshold) .& (varr[2:Int(params.nt), :] .> params.threshold)
    return Bool.(fire)
end

@kwdef mutable struct SimulationParams
    Ne::Int = 100              # The number of excitatory neuron.
    Ni::Int = 0                # The number of inhibitory neuron.
    NiC::Int = 30              # The number of excitatory connection.
    NeC::Int = 1               # The number of inhibitory connection.
    NiE::Int = 100             # The number of excitatory neuron to input stimulus.
    T::Float64 = 1600.0        # Time span [msec].
    dt::Float64 = 0.05         # Time step [msec].
    V_exc::Float64 = 0.0       # Excitatory Voltage.
    Er::Float64 = -70.0        # Resting Potential.
    V_inh::Float64 = -65.0     # Inhibitory Voltage.
    threshold::Float64 = -40.0  # Threshold of fire.
    nt::Int = 0                # The number of time step.
    d::Any = Normal(1.0, 0.5)  # Random set of normal distribution.
end

function SimulationParams(; Ne, Ni, NiC, NeC, NiE, T, dt, V_exc, Er, V_inh, threshold, d)
    nt = Int(T / dt)  # nt を T/dt で計算
    return SimulationParams(Ne, Ni, NiC, NeC, NiE, T, dt, V_exc, Er, V_inh, threshold, nt, d)
end

params = SimulationParams(
    Ne=100,
    Ni=0,
    NeC=1,
    NiC=30,
    NiE=100,
    T=1600.0,
    dt=0.05,
    V_exc=.0,
    Er=-65.0,
    V_inh=-70.0,
    threshold=-30.0,
    d=Normal(1.0, 0.5)
)

function alpha_generate(alpha, tonic, phasic, Ne, Ni, data_length)
    neurons = Int(Ne + Ni)
    alpha_tonic = Int(alpha * neurons)
    alpha_phasic = neurons - alpha_tonic
    noise_spikes = isone.(zeros(data_length, Ne+Ni))
    @inbounds for j = 1:1:alpha_tonic
        noise_spikes[:,j] = tonic[:,j]
    end
    @inbounds for j = 1:1:alpha_phasic
        noise_spikes[:,j] = phasic[:,j]
    end
    return noise_spikes
end

function main()
    # ---------------------- #
    # コマンドラインの設定
    # ---------------------- #
    # 引数の設定
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nec", "-a"
            help = "The number of excitatory connection"
            arg_type = Int64
            default = 1
        "--nic", "-b"
            help = "The number of inhibitory connection"
            arg_type = Int64
            default = 30
        "--fexc", "-e"
            help = "Base Input Frequency [Hz]"
            arg_type = Float64
            default = 100.0
            # required = true # 必須の引数
        "--fdiff", "-d"
            help = "Stimulus change frequency [Hz]"
            arg_type = Float64
            default = 200.0
            # required = true # 必須の引数
        "--finh", "-i"
            help = "Noise Frequency [Hz]"
            arg_type = Float64
            default = 100.0
            # required = true # 必須の引数
        "--width", "-w"
            help = "Width of stimulus change [msec]"
            arg_type = Float64
            default = 10.0
        "--output_dir", "-o"
            help = "Output Folder"
            default = "outputs" # デフォルト値
        # 1.0: Tonic, 0.0: Phasic
        "--rate", "-r"
            help = "Tonic or Phasic Rate"
            arg_type = Float64
            default = 1.0
            # action = :store_true # フラグ引数
        "--delta", "-s"
            help = "Synaptic depression coefficient"
            arg_type = Float64
            default = 0.5
    end

    # コマンドライン引数をパース
    args = parse_args(s)

    println(args)

    # パラメータの上書き
    params.NeC = args["nec"]
    params.NiC = args["nic"]
    SIGNAL_WIDTH = args["width"]

    # ------------------------------- #
    # ディレクトリの作成とパラメータの保存
    # ------------------------------- #
    dir_path = args["output_dir"]

    if !isdir(dir_path)
        println("Directory does not exist. Creating: $dir_path")
        mkpath(dir_path)
    else
        println("Directory already exists: $dir_path")
    end

    # 指定したパラメータの保存(.json)
    open("$dir_path/args.json", "w") do file
        JSON.print(file, args, 4)
    end
    open("$dir_path/params.json", "w") do file
        JSON.print(file, params, 4)
    end
    # params = JSON.parsefile("params.json")  # In case of loading json file.

    # ---------------------- #
    # Main Routine
    # ---------------------- #
    run_path=100
    neuron_count = params.Ne + params.Ni
    # 配列の数
    num_layers = 5
    # すべての配列を一つのリストにまとめる
    fires = [zeros(Bool, params.nt, neuron_count, run_path) for _ in 1:num_layers]
    dt_signal, num_digits = params.dt*1e-3, 5;

    @time Threads.@threads for j = 1:1:run_path
        # 変動シード
        #rng = MersenneTwister(rand(UInt) + Threads.threadid())  # スレッドIDを利用して異なる初期シードを設定
        #seed = rand(rng, 1:10^6)  
        #Random.seed!(rng, seed)
        #println("Thread $(Threads.threadid()), Seed: $seed, Rand: ", rand(rng))
        
        # 固定シード
        base_seed = 42  # グローバルに固定シードを決める
        thread_seed = base_seed + Threads.threadid()  # スレッドごとに異なるシード
        rng = MersenneTwister(thread_seed)  # 各スレッドで決定論的に異なる RNG を作る
        Random.seed!(rng, thread_seed)

        freq = InputFreqSetting(
            Fs=1/params.dt,
            StopTime=params.T,
            Width=SIGNAL_WIDTH,
            f_diff=args["fdiff"],
            f_exc=args["fexc"],
            mid_point=800.0,
            f_inh=args["finh"]
        );
        inputs, currents, noises = step_freq_response(freq);
    
        pre_spikes, b_tonic = GenerateEvents.generate_events_tonic(inputs, noises, dt_signal, neuron_count)
        _, b_phasic =GenerateEvents.generate_events_phasic(inputs, dt_signal, neuron_count, args["finh"], params.T, params.dt)
        b_spikes = alpha_generate(args["rate"], b_tonic, b_phasic, params.Ne, params.Ni, length(inputs))
        fires[1][1:end-1,:,j] = intra_layer(length(inputs), pre_spikes, b_spikes, params, args["delta"])

        _, b_tonic = GenerateEvents.generate_events_tonic(inputs, noises, dt_signal, neuron_count)
        _, b_phasic = GenerateEvents.generate_events_phasic(inputs, dt_signal, neuron_count, args["finh"], params.T, params.dt)
        b_spikes = alpha_generate(args["rate"], b_tonic, b_phasic, params.Ne, params.Ni, length(inputs))
        fires[2][1:end-1,:,j] = between_layer(length(inputs), fires[1][:,:,j], b_spikes, params, args["delta"])
        
        _, b_tonic = GenerateEvents.generate_events_tonic(inputs, noises, dt_signal, neuron_count)
        _, b_phasic = GenerateEvents.generate_events_phasic(inputs, dt_signal, neuron_count, args["finh"], params.T, params.dt)
        b_spikes = alpha_generate(args["rate"], b_tonic, b_phasic, params.Ne, params.Ni, length(inputs))
        fires[3][1:end-1,:,j] = between_layer(length(inputs), fires[2][:,:,j], b_spikes, params, args["delta"])

        _, b_tonic = GenerateEvents.generate_events_tonic(inputs, noises, dt_signal, neuron_count)
        _, b_phasic = GenerateEvents.generate_events_phasic(inputs, dt_signal, neuron_count, args["finh"], params.T, params.dt)
        b_spikes = alpha_generate(args["rate"], b_tonic, b_phasic, params.Ne, params.Ni, length(inputs))
        fires[4][1:end-1,:,j] = between_layer(length(inputs), fires[3][:,:,j], b_spikes, params, args["delta"])

        _, b_tonic = GenerateEvents.generate_events_tonic(inputs, noises, dt_signal, neuron_count)
        _, b_phasic = GenerateEvents.generate_events_phasic(inputs, dt_signal, neuron_count, args["finh"], params.T, params.dt)
        b_spikes = alpha_generate(args["rate"], b_tonic, b_phasic, params.Ne, params.Ni, length(inputs))
        fires[5][1:end-1,:,j] = between_layer(length(inputs), fires[4][:,:,j], b_spikes, params, args["delta"])
    end

    # --------------------------------- #
    # Save firing events in each layer.
    # --------------------------------- #
    for i = 1:num_layers
        indices_list = findall(x -> x == true, fires[i])
        # バイナリファイルに保存
        open("$dir_path/fire$i.bin", "w") do file
            serialize(file, indices_list)
        end
    end
end

# 実行
main()
