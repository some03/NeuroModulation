julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 10 --output outputs/tonic_width_010 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 30 --output outputs/tonic_width_030 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 50 --output outputs/tonic_width_050 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 100 --output outputs/tonic_width_100 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 150 --output outputs/tonic_width_150 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 200 --output outputs/tonic_width_200 --rate 1.0

julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 10 --output outputs/phasic_width_010 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 30 --output outputs/phasic_width_030 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 50 --output outputs/phasic_width_050 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 100 --output outputs/phasic_width_100 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 150 --output outputs/phasic_width_150 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 200 --output outputs/phasic_width_200 --rate 0