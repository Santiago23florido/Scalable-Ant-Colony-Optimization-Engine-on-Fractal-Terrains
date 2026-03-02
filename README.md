# Scalable Ant Colony Optimization Engine on Fractal Terrains

This project implements an Ant Colony Optimization (ACO) simulation on a fractal terrain, with profiling support to measure performance per simulation iteration.

## Repository Layout

- `nonvectorized/`: baseline implementation (object-per-ant design, SDL rendering, CSV timing export)
- `Subject.pdf`: project requirements and optimization roadmap (vectorization, OpenMP, MPI)

## Baseline Features

- Real-time SDL visualization of ants, pheromones, nest, and food
- Full simulation loop with pheromone update and evaporation
- Iteration-level timing breakdown exported to CSV
- Automatic summary export with key metadata (including first-food iteration)

## Requirements (Linux)

- `g++` with C++17 support
- `make`
- `libsdl2-dev`

Install dependencies:

```bash
sudo apt update
sudo apt install -y build-essential libsdl2-dev
```

## Build

From the `nonvectorized` folder:

```bash
cd nonvectorized
make clean
make all CXXFLAGS2='-std=c++17 -O2 -march=native -Wall'
```

## Run (Default: Full Simulation + Render + CSV Logging)

```bash
./ant_simu.exe
```

By default, the program:

- opens the SDL window and runs until you close it
- appends per-iteration timings to `nonvectorized/results/iter.csv`
- writes aggregated metrics to `nonvectorized/results/summary.csv` when the run ends

## Optional CLI Flags

```bash
./ant_simu.exe --help
```

Main options:

- `--headless`: disable rendering/event polling timings
- `--max-iterations N`: stop automatically after `N` iterations
- `--warmup-iterations N`: skip first `N` iterations from stats
- `--timing-csv PATH`: custom per-iteration output file
- `--summary-csv PATH`: custom summary output file

## Output Files

### `results/iter.csv`

Per-iteration values:

- `ants_advance_ms`
- `evaporation_ms`
- `update_ms`
- `advance_total_ms`
- `render_ms`
- `blit_ms`
- `iteration_total_ms`
- `food_quantity`

### `results/summary.csv`

Aggregated metrics (`count`, `total_ms`, `mean_ms`, `min_ms`, `max_ms`) + metadata:

- `total_iterations`
- `measured_iterations`
- `final_food_quantity`
- `first_food_iteration` (or `not_reached`)

## Next Optimization Stages

Following `Subject.pdf`, this baseline is intended to be compared against:

1. vectorized data-oriented implementation
2. shared-memory parallel version (OpenMP)
3. distributed-memory parallel version (MPI strategies)
