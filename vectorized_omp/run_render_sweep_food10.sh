#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

make all

RESULTS_DIR="results/sweeps_render_food10"
mkdir -p "$RESULTS_DIR"

ANTS_LIST=(5000 10000 20000 40000 80000 160000)
THREADS_LIST=(2 3 4 5 6 7 8 9 10 11 12)

for ants in "${ANTS_LIST[@]}"; do
  for threads in "${THREADS_LIST[@]}"; do
    iter_csv="$RESULTS_DIR/iter_threads${threads}_ants${ants}.csv"
    summary_csv="$RESULTS_DIR/summary_threads${threads}_ants${ants}.csv"

    echo "Running vectorized_omp simulation: threads=${threads}, ants=${ants}"
    OMP_NUM_THREADS="$threads" ./ant_simu.exe \
      --nb-ants "$ants" \
      --post-first-food-iterations 10 \
      --timing-csv "$iter_csv" \
      --summary-csv "$summary_csv"
  done
done

echo "Done. Results saved in: $RESULTS_DIR"
