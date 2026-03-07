# Distributed Subdomain Hybrid MPI+OpenMP

This folder is a copy of `distributed_subdomain_mpi` adapted to run in **hybrid mode**:
- MPI for 2D subdomain decomposition and inter-rank communication.
- OpenMP inside each MPI rank for intra-subdomain compute.

## Build

```bash
cmake -S . -B cmake-build-hybrid -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-hybrid -j
```

Binary:
- `build/distributed_subdomain_hybrid_mpi_omp_sim`

## Sweep Execution

Run the provided sweep script:

```bash
./run_render_sweep_food10.sh
```

The sweep uses:
- MPI ranks: `2..8`
- OpenMP threads per rank: `2, 3, 4`
- Ant counts: `5000, 10000, 20000, 40000, 80000, 160000`

Generated CSV naming:
- `iter_ranks{R}_omp{T}_ants{N}.csv`
- `summary_ranks{R}_omp{T}_ants{N}.csv`

Output directory:
- `results/sweeps_render_food10/`

## Plot Hybrid Sweep

```bash
./plot_sweep_times.py
```

Generated figures:
- `hybrid_total_breakdown_omp{T}_ants{N}.png`
- `hybrid_calc_only_omp{T}_ants{N}.png`
- `hybrid_calc_speedup_omp{T}_ants{N}.png`

Output directory:
- `results/sweeps_render_food10/plots`
