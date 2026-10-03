# Attempt to reproduce the supplied frequency-change curves using the saved
# notebook's historical Xp calculation. These error bars are NOT trial uncertainty.
include(joinpath(@__DIR__, "plot_activity.jl"))
using Statistics

function historical_trace(run, events)
    isapprox(run.dt, 0.05; atol=1e-9) || error("Historical Xp expects dt=0.05 ms")
    counts = zeros(Int, run.nt + 1)
    for event in events
        counts[event[1]] += 1
    end
    # The notebook uses inclusive 101-step windows spaced 100 steps apart.
    # Use the recorded simulation length, rather than the last spike time.
    trace = zeros(Float32, fld(run.nt, 100))
    for i in eachindex(trace)
        center = (i - 1) * 100 + 1
        lo = max(center - 50, 1)
        hi = min(center + 50, length(counts))
        trace[i] = sum(@view counts[lo:hi]) / run.trials
    end
    return trace
end

function historical_metric(trace)
    length(trace) >= 220 || error("Historical Xp requires a recording through 1100 ms")
    # Preserves the saved notebook's 41-bin sum / 40 baseline and zero padding.
    baseline = sum(trace[180:220]) / 40
    target = trace[140:180] .- baseline
    padded = zeros(Float32, length(trace))
    for i in 1:40
        lo = max(1, i - 1)
        padded[i] = sum(@view target[lo:(i + 1)])
    end
    return maximum(padded), std(padded)
end

function plot_frequency_change()
    length(ARGS) == 1 || error("Usage: plot_fdiff.jl SWEEP_OUTPUT_ROOT")
    root = ARGS[1]
    isdir(root) || error("Output folder not found: $root")
    candidates = NamedTuple[]
    for name in readdir(root)
        matched = match(r"^(tonic|phasic)_fdiff_(\d+)$", name)
        matched === nothing && continue
        run = load_run(joinpath(root, name))
        mode = matched.captures[1]
        rate = Float64(run.args["rate"])
        isapprox(rate, mode == "tonic" ? 1.0 : 0.0) || error("Folder and rate disagree: $name")
        push!(candidates, (mode=mode, fdiff=Float64(run.args["fdiff"]), run=run))
    end
    isempty(candidates) && error("No tonic_fdiff_* or phasic_fdiff_* outputs found")
    figures = joinpath(root, "figures")
    mkpath(figures)
    for layer in 1:5
        p = plot(; xlabel="Frequency change (Hz)", ylabel="Xp (historical calculation)",
                 title="Layer $layer", size=(750, 500), legend=:topleft)
        open(joinpath(figures, "fdiff_layer$(layer).csv"), "w") do io
            println(io, "mode,fdiff_hz,xp,historical_time_std,trials,source_directory")
            for mode in ("tonic", "phasic")
                cases = sort(filter(c -> c.mode == mode, candidates); by=c -> c.fdiff)
                isempty(cases) && continue
                x, y, spread = Float64[], Float64[], Float64[]
                for case in cases
                    events = read_events(case.run, layer)
                    peak, time_std = historical_metric(historical_trace(case.run, events))
                    push!(x, case.fdiff)
                    push!(y, peak)
                    push!(spread, time_std)
                    println(io, "$(mode),$(case.fdiff),$(peak),$(time_std),$(case.run.trials),$(basename(case.run.folder))")
                end
                plot!(p, x, y; yerr=spread, color=mode == "tonic" ? :blue : :red,
                      marker=mode == "tonic" ? :circle : :rect, label=uppercasefirst(mode))
            end
        end
        savefig(p, joinpath(figures, "Fdiff_layer$(layer).pdf"))
        savefig(p, joinpath(figures, "Fdiff_layer$(layer).png"))
    end
    println("Saved: ", abspath(figures))
    println("Historical error bars are time-series standard deviations, not trial uncertainty.")
end

plot_frequency_change()