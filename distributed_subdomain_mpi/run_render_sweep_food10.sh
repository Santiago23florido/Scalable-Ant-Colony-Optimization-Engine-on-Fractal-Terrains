#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cmake -S . -B cmake-build -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build -j

RESULTS_DIR="results/sweeps_render_food10"
mkdir -p "$RESULTS_DIR"

ANTS_LIST=(5000 10000 20000 40000 80000 160000)
THREADS_LIST=(2 3 4 5 6 7 8 9 10 11 12)

# By default allow launching more MPI ranks than available slots/cores.
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
MPIRUN_EXTRA_ARGS_STR="${MPIRUN_EXTRA_ARGS:---oversubscribe}"
read -r -a MPIRUN_EXTRA_ARGS <<< "$MPIRUN_EXTRA_ARGS_STR"

for ants in "${ANTS_LIST[@]}"; do
  for threads in "${THREADS_LIST[@]}"; do
    iter_csv="$RESULTS_DIR/iter_threads${threads}_ants${ants}.csv"
    summary_csv="$RESULTS_DIR/summary_threads${threads}_ants${ants}.csv"

    echo "Running distributed simulation: mpi_ranks=${threads}, ants=${ants}"
    "$MPIRUN_BIN" "${MPIRUN_EXTRA_ARGS[@]}" -np "$threads" ./build/distributed_subdomain_mpi_sim \
      --nb-ants "$ants" \
      --post-first-food-iterations 10 \
      --timing-csv "$iter_csv" \
      --summary-csv "$summary_csv"
  done
done

echo "Done. Results saved in: $RESULTS_DIR"
