#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_DIR="cmake-build-hybrid"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j

RESULTS_DIR="results/sweeps_render_food10"
mkdir -p "$RESULTS_DIR"

ANTS_LIST=(5000 10000 20000 40000 80000 160000)
MPI_RANKS_LIST=(2 3 4 5 6 7 8)
OMP_THREADS_LIST=(2 3 4)

# By default allow launching more MPI ranks than available slots/cores.
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
MPIRUN_EXTRA_ARGS_STR="${MPIRUN_EXTRA_ARGS:---oversubscribe}"
read -r -a MPIRUN_EXTRA_ARGS <<< "$MPIRUN_EXTRA_ARGS_STR"

for ants in "${ANTS_LIST[@]}"; do
  for omp_threads in "${OMP_THREADS_LIST[@]}"; do
    for mpi_ranks in "${MPI_RANKS_LIST[@]}"; do
      iter_csv="$RESULTS_DIR/iter_ranks${mpi_ranks}_omp${omp_threads}_ants${ants}.csv"
      summary_csv="$RESULTS_DIR/summary_ranks${mpi_ranks}_omp${omp_threads}_ants${ants}.csv"

      echo "Running hybrid simulation: mpi_ranks=${mpi_ranks}, omp_threads=${omp_threads}, ants=${ants}"
      OMP_NUM_THREADS="$omp_threads" \
      "$MPIRUN_BIN" "${MPIRUN_EXTRA_ARGS[@]}" -np "$mpi_ranks" ./build/distributed_subdomain_hybrid_mpi_omp_sim \
        --nb-ants "$ants" \
        --post-first-food-iterations 10 \
        --timing-csv "$iter_csv" \
        --summary-csv "$summary_csv"
    done
  done
done

echo "Done. Results saved in: $RESULTS_DIR"
