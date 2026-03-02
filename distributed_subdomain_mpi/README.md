# Distributed Subdomain MPI Plan (Second Approach)

## Folder Name
Use this folder for the project:

- `distributed_subdomain_mpi`

## Objective
Implement the **second distributed-memory strategy**:

- each MPI process stores only a subdomain of the map
- each process manages only ants currently inside its subdomain
- processes exchange only border (ghost) data and migrating ants

No production code is included here, only implementation guidance in pseudocode/text.

## Base Project to Fork
Start from `vectorized/` as base, then copy structure into this folder:

- `include/`
- `src/`
- `build/`
- `results/`

## Files to Modify (Existing)

1. `src/ant_simu.cpp`
- add MPI initialization/finalization flow
- replace single-domain setup with subdomain decomposition setup
- replace global loop with local-step + communications + migration loop
- add global reductions for metrics (`food_quantity`, timing)

2. `include/pheronome.hpp`
- change map ownership from global map to **local map + ghost border**
- expose local/ghost indexing helpers
- split border exchanges for pheromone channels

3. `src/ant_system.cpp` and `include/ant_system.hpp`
- update ant advancement to detect out-of-subdomain moves
- queue outbound ants by destination rank
- remove local ants that migrated out
- insert received ants from neighbors

4. `src/renderer.cpp` and rendering usage in `src/ant_simu.cpp`
- keep rendering only on rank 0 (or disable in MPI runs)
- if rendering is kept, gather a lightweight visualization payload on rank 0

5. `README.md` (inside this folder)
- document domain decomposition, communication pattern, and benchmark protocol

## Files to Add (New)

1. `include/domain_partition.hpp` + `src/domain_partition.cpp`
- define domain decomposition and rank-neighbor metadata

2. `include/halo_exchange.hpp` + `src/halo_exchange.cpp`
- isolate ghost-border communication for pheromones

3. `include/ant_migration.hpp` + `src/ant_migration.cpp`
- isolate ant packing/unpacking and transfer routines

4. `include/mpi_metrics.hpp` + `src/mpi_metrics.cpp`
- aggregate timing and compute speedup tables

## Required Data Model Changes

## Domain decomposition
Use a 2D Cartesian process grid if possible.

Pseudocode:

```text
create_cartesian_topology(world_size)
compute_local_bounds(rank) -> [x0:x1), [y0:y1)
identify_neighbors(rank) -> left,right,up,down
```

## Local pheromone storage
Each process stores:

- interior cells for local subdomain
- 1-cell ghost layer on each side

Pseudocode:

```text
allocate local_v1[(nx+2)*(ny+2)]
allocate local_v2[(nx+2)*(ny+2)]
allocate local_buf_v1[(nx+2)*(ny+2)]
allocate local_buf_v2[(nx+2)*(ny+2)]
```

## Ant ownership rule
An ant belongs to the process whose interior cell contains its position.

Pseudocode:

```text
owner_rank = rank_of(global_x, global_y)
if owner_rank != my_rank: ant must migrate
```

## Iteration Loop (Core Algorithm)

Pseudocode:

```text
for each iteration:
  start timer iteration_total

  1) exchange pheromone ghost borders (v1, v2)

  2) local ant advance:
       for each local ant:
         while consumed_time < 1:
           read neighbor pheromone values (interior + ghost)
           choose move (explore/exploit)
           if target cell outside local interior:
             pack ant into outbound_buffer[target_rank]
             mark ant for removal from local list
             break
           else:
             apply local pheromone marking
             update ant state (food/nest)

  3) compact local ant arrays (remove migrated ants)

  4) exchange outbound ants with neighbor ranks

  5) append inbound ants to local ant arrays

  6) local evaporation and local update of pheromones

  7) local timing accumulation

  8) every K iterations or at end:
       MPI_Reduce / MPI_Allreduce for food counter and timing summaries

end loop
```

## Communication Strategy

## Halo exchange (pheromones)
Use non-blocking sends/receives with one message per direction per channel (or packed channels).

Pseudocode:

```text
post Irecv for left/right/up/down ghost lines
post Isend for interior border lines
waitall
```

## Ant migration exchange
Use two-phase communication to avoid variable-size ambiguity.

Pseudocode:

```text
phase A: exchange outbound counts with neighbors
phase B: send/recv packed ant payloads using received counts
unpack incoming ants into local arrays
```

## Load Imbalance Handling

Because nest-centered regions may hold more ants:

- monitor local ant counts periodically
- report imbalance ratio = max(local_ants) / mean(local_ants)
- optional: periodic repartition trigger only if imbalance exceeds threshold

Pseudocode:

```text
if iteration % monitor_period == 0:
  gather local_ant_count
  compute imbalance_ratio
  if imbalance_ratio > threshold:
    flag potential repartition (optional advanced step)
```

## Functions That Must Change (Behavior)

1. `advance_all(...)` in `AntSystem`
- before: assumes all ants stay local
- after: returns outbound migration buffers and local food increment

2. pheromone accessors in `pheronome`
- before: global flat map semantics
- after: local interior + ghost semantics with local index helpers

3. main simulation loop in `ant_simu.cpp`
- before: single-process sequential iteration
- after: distributed iteration with halo exchange, migration, and reductions

4. summary export
- before: local single-run summary
- after: rank 0 writes global summary, includes process count and speedup

## Benchmark and Acceleration Table

Run with process counts:

- `p = 1, 2, 4, 8, ...`

Measure:

- `T_p = mean(iteration_total_ms)` from global aggregated stats

Compute:

- `speedup(p) = T_1 / T_p`
- `efficiency(p) = speedup(p) / p`

Output table columns:

- `processes`
- `mean_iteration_total_ms`
- `speedup`
- `efficiency`
- `mean_ants_advance_ms`
- `mean_exchange_ms`

## Acceptance Checklist

- local memory scales with subdomain size, not full map size
- only border halos and migrating ants are communicated
- global dynamics remain consistent with baseline behavior
- rank 0 writes aggregated `summary.csv` and speedup table
- results reported for multiple process counts
