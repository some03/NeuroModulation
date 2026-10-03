
# Usage

```
$ julia --project=PTProject/Project.toml examples/synapse_sample.jl

# 並列処理を入れないと実行時間がかなりかかる。
$ julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl --output outputs --rate 0.7

# width(rate=1.0と0.0で確認して、識別できるようにtonic phasicをつける必要があるので注意)
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 5 --output outputs/tonic_width_005 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 10 --output outputs/tonic_width_010 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 30 --output outputs/tonic_width_030 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 50 --output outputs/tonic_width_050 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 70 --output outputs/tonic_width_070 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 100 --output outputs/tonic_width_100 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 150 --output outputs/tonic_width_150 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 200 --output outputs/tonic_width_200 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 300 --output outputs/tonic_width_300 --rate 1.0

julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 5 --output outputs/phasic_width_005 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 10 --output outputs/phasic_width_010 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 30 --output outputs/phasic_width_030 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 50 --output outputs/phasic_width_050 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 70 --output outputs/phasic_width_070 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 100 --output outputs/phasic_width_100 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 150 --output outputs/phasic_width_150 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 200 --output outputs/phasic_width_200 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -w 300 --output outputs/phasic_width_300 --rate 0

# f_diff
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 100 --output outputs/tonic_fdiff_100 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 200 --output outputs/tonic_fdiff_200 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 300 --output outputs/tonic_fdiff_300 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 400 --output outputs/tonic_fdiff_400 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 500 --output outputs/tonic_fdiff_500 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 600 --output outputs/tonic_fdiff_600 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 700 --output outputs/tonic_fdiff_700 --rate 1.0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 800 --output outputs/tonic_fdiff_800 --rate 1.0

julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 100 --output outputs/phasic_fdiff_100 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 200 --output outputs/phasic_fdiff_200 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 300 --output outputs/phasic_fdiff_300 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 400 --output outputs/phasic_fdiff_400 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 500 --output outputs/phasic_fdiff_500 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 600 --output outputs/phasic_fdiff_600 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 700 --output outputs/phasic_fdiff_700 --rate 0
julia --threads 4 --project=PTProject/Project.toml pipelines/main.jl -d 800 --output outputs/phasic_fdiff_800 --rate 0
```