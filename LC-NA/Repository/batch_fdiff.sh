julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 100 --output outputs/tonic_fdiff_100 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 200 --output outputs/tonic_fdiff_200 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 300 --output outputs/tonic_fdiff_300 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 400 --output outputs/tonic_fdiff_400 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 500 --output outputs/tonic_fdiff_500 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 600 --output outputs/tonic_fdiff_600 --rate 1.0

julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 100 --output outputs/phasic_fdiff_100 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 200 --output outputs/phasic_fdiff_200 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 300 --output outputs/phasic_fdiff_300 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 400 --output outputs/phasic_fdiff_400 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 500 --output outputs/phasic_fdiff_500 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 600 --output outputs/phasic_fdiff_600 --rate 0