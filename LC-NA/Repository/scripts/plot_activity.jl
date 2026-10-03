# Run from the Repository folder:
# julia +1.11.2 --project=PTProject scripts/plot_activity.jl outputs/smoke_tonic outputs/smoke_phasic
ENV["GKSwstype"] = "100"
using Serialization, JSON, Plots

function load_run(folder)
    params = JSON.parsefile(joinpath(folder, "params.json"))
    args = JSON.parsefile(joinpath(folder, "args.json"))
    return (
        folder=folder,
        dt=Float64(params["dt"]),
        nt=Int(params["nt"]),
        neurons=Int(params["Ne"] + params["Ni"]),
        trials=Int(get(args, "trials", 100)),
        args=args,
    )
end

function read_events(run, layer)
    path = joinpath(run.folder, "fire$(layer).bin")
    events = open(deserialize, path)
    for event in events
        t, neuron, trial = Tuple(event)
        1 <= t <= run.nt || error("Time index outside params.json: $path")
        1 <= neuron <= run.neurons || error("Neuron index outside params.json: $path")
        1 <= trial <= run.trials || error("Trial index outside args.json: $path")
    end
    return events
end

function binned_activity(run, events; bin_ms=5.0)
    steps = round(Int, bin_ms / run.dt)
    steps >= 1 || error("Time bin must be at least one simulation step")
    isapprox(steps * run.dt, bin_ms; atol=1e-8) || error("Time bin must divide dt exactly")
    bins = cld(run.nt, steps)
    counts = zeros(Float64, bins)
    for event in events
        counts[fld(event[1] - 1, steps) + 1] += 1
    end
    times = ((0:(bins - 1)) .+ 0.5) .* bin_ms
    return times, counts ./ run.trials
end

function main()
    2 <= length(ARGS) <= 3 || error("Usage: plot_activity.jl TONIC_DIR PHASIC_DIR [FIGURE_DIR]")
    tonic, phasic = load_run(ARGS[1]), load_run(ARGS[2])
    output_dir = length(ARGS) == 3 ? ARGS[3] : joinpath("outputs", "smoke_figures")
    mkpath(output_dir)
    isapprox(tonic.dt, phasic.dt) && tonic.nt == phasic.nt || error("Time settings differ")
    panels = []
    rasters = []
    for layer in 1:5
        tonic_events = read_events(tonic, layer)
        phasic_events = read_events(phasic, layer)
        times, tonic_counts = binned_activity(tonic, tonic_events)
        _, phasic_counts = binned_activity(phasic, phasic_events)
        p = plot(times, tonic_counts; color=:blue, label="Tonic", title="Layer $layer",
                 ylabel="Spikes / trial / 5 ms", xlabel="Time (ms)", legend=:topright)
        plot!(p, times, phasic_counts; color=:red, label="Phasic")
        vline!(p, [800.0]; color=:gray, linestyle=:dash, label="")
        push!(panels, p)
        open(joinpath(output_dir, "activity_layer$(layer).csv"), "w") do io
            println(io, "time_ms,tonic_spikes_per_trial,phasic_spikes_per_trial")
            for i in eachindex(times)
                println(io, "$(times[i]),$(tonic_counts[i]),$(phasic_counts[i])")
            end
        end
        for (run, events, label) in ((tonic, tonic_events, "Tonic"), (phasic, phasic_events, "Phasic"))
            first_trial = filter(event -> event[3] == 1, events)
            spike_times = Float64[(event[1] - 1) * run.dt for event in first_trial]
            neuron_ids = Int[event[2] for event in first_trial]
            raster = scatter(spike_times, neuron_ids; markersize=1, color=:black, legend=false,
                             title="$label, layer $layer, trial 1", xlabel="Time (ms)",
                             ylabel="Neuron", xlims=(0, run.nt * run.dt), ylims=(0, run.neurons + 1))
            push!(rasters, raster)
        end
    end
    activity = plot(panels...; layout=(5, 1), size=(1100, 1500))
    savefig(activity, joinpath(output_dir, "activity.png"))
    raster_plot = plot(rasters...; layout=(5, 2), size=(1300, 1500))
    savefig(raster_plot, joinpath(output_dir, "raster.png"))
    println("Saved: ", abspath(output_dir))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end