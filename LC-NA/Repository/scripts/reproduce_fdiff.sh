#!/usr/bin/env bash
set -euo pipefail

lc_trials="${1:-100}"
if [[ ! "$lc_trials" =~ ^[0-9]+$ ]] || ((lc_trials < 1)); then
    printf '%s\n' 'Trial count must be a positive integer.' >&2
    exit 1
fi
lc_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$lc_repo"
lc_root="${2:-outputs/reproduce_n${lc_trials}}"
lc_version="${LC_JULIA_VERSION:-1.11.2}"
lc_threads="${LC_JULIA_THREADS:-4}"

for lc_fdiff in 100 200 300 400 500 600 700 800; do
    for lc_mode in tonic phasic; do
        lc_rate=1.0
        if [[ "$lc_mode" == phasic ]]; then lc_rate=0.0; fi
        lc_target="$lc_root/${lc_mode}_fdiff_${lc_fdiff}"
        if [[ -e "$lc_target" ]]; then
            printf 'Output already exists: %s. Choose a new output root.\n' "$lc_target" >&2
            exit 1
        fi
        GKSwstype=100 julia "+$lc_version" --threads "$lc_threads" --project=PTProject \
            pipelines/main.jl --trials "$lc_trials" --rate "$lc_rate" \
            --fexc 100 --fdiff "$lc_fdiff" --finh 100 --width 10 --delta 0.5 \
            --output_dir "$lc_target"
    done
done
julia "+$lc_version" --project=PTProject scripts/plot_fdiff.jl "$lc_root"