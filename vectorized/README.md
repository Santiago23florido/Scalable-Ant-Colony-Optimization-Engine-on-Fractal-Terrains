# Vectorized Ant Colony Simulation

This directory contains a vectorized refactor of the baseline ACO simulation.

## What Was Vectorized

The original `ant` object-per-instance model was replaced by a structure-of-arrays (`AntSystem`) in `include/ant_system.hpp` and `src/ant_system.cpp`.

The ant state is now stored in contiguous arrays:

- `m_pos_x`: x position for each ant
- `m_pos_y`: y position for each ant
- `m_state`: loaded/unloaded flag for each ant
- `m_seed`: RNG state for each ant

An ant is identified by its index across these arrays.

The pheromone map was also converted to a structure-of-arrays layout:

- `v1` and `v2` are stored in separate contiguous buffers
- evaporation/update use separate `buf_v1` and `buf_v2`
- hot-path accesses use flat indices (`idx`, `idx±1`, `idx±stride`)
- pheromone marking uses flat-index neighbors (no temporary point objects in the hot path)
- ant updates keep `x/y/state/seed` in local variables and write back once per ant step loop

## Why This Improves Performance

- Better cache locality when iterating over all ants
- Lower per-ant object overhead
- Better memory locality for pheromone channel reads/writes
- Fewer temporary objects and less random access indirection in the hot loop
- Data layout ready for future SIMD and OpenMP work
- Same algorithmic behavior as baseline, so timing comparisons remain valid

## Current Performance Note

This refactor is cache-friendlier and usually improves `evaporation` and often `advance_total`.
`ants_advance` may still be close to baseline because the ant movement loop is branch-heavy and RNG-driven, which limits auto-vectorization.

## Behavioral Equivalence

The movement and pheromone logic was kept equivalent to the baseline:

- same random exploration/exploitation rule
- same consumed-time loop per ant
- same pheromone marking, evaporation, and update order
- same nest/food state transitions and food counter updates

## Project Structure

- `include/`: headers
- `src/`: implementation files
- `build/`: object files
- `results/`: timing outputs (`iter.csv`, `summary.csv`)

## Build

```bash
cd vectorized
make clean
make all
```

If needed:

```bash
make all CXXFLAGS2='-std=c++17 -Iinclude -O2 -march=native -Wall'
```

## Run

Default run (render + continuous CSV logging):

```bash
./ant_simu.exe
```

Headless timed run:

```bash
./ant_simu.exe --headless --max-iterations 1000 --warmup-iterations 100
```

## Timing Output

Per-iteration file:

- `results/iter.csv`

Summary file:

- `results/summary.csv`

`summary.csv` includes:

- mean/min/max timing per measured section
- `total_iterations`
- `measured_iterations`
- `final_food_quantity`
- `first_food_iteration` (or `not_reached`)

## Comparison Workflow

1. Run `nonvectorized` with the same machine conditions and iteration budget.
2. Run `vectorized` with the same flags.
3. Compare `ants_advance` and `iteration_total` mean times from both summaries.
4. Compute speedup as `baseline_time / vectorized_time`.
