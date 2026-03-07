#include "ant_migration.hpp"

#include <algorithm>
#include <array>

#include "rand_generator.hpp"

namespace mpi_subdomain {

namespace {
constexpr int dir_count = 4;
constexpr int dir_up = 0;
constexpr int dir_down = 1;
constexpr int dir_left = 2;
constexpr int dir_right = 3;

// Checks if a global x coordinate is inside the global map.
inline bool in_global_x(const DomainDecomposition& decomp, int gx) {
    return gx >= 0 && gx < decomp.global_nx;
}

// Checks if a global y coordinate is inside the global map.
inline bool in_global_y(const DomainDecomposition& decomp, int gy) {
    return gy >= 0 && gy < decomp.global_ny;
}

// Reads pheromone at global coordinates using local halo-padded storage.
inline double read_pheromone(const DomainDecomposition& decomp, const std::vector<double>& field, int gx, int gy) {
    if (!in_global_x(decomp, gx) || !in_global_y(decomp, gy)) {
        return -1.0;
    }

    const int lx = decomp.local_x_from_global(gx);
    const int ly = decomp.local_y_from_global(gy);
    if (lx < 0 || lx >= decomp.stride_x() || ly < 0 || ly >= decomp.stride_y()) {
        return -1.0;
    }

    return field[decomp.idx(lx, ly)];
}

// Reads terrain movement cost at global coordinates using local halo-padded storage.
inline double read_terrain(const DomainDecomposition& decomp, const std::vector<double>& terrain, int gx, int gy) {
    if (!in_global_x(decomp, gx) || !in_global_y(decomp, gy)) {
        return 1.0;
    }

    const int lx = decomp.local_x_from_global(gx);
    const int ly = decomp.local_y_from_global(gy);
    if (lx < 0 || lx >= decomp.stride_x() || ly < 0 || ly >= decomp.stride_y()) {
        return 1.0;
    }

    return terrain[decomp.idx(lx, ly)];
}

// Updates next pheromone values for one owned global cell.
inline void mark_cell_from_current(const DomainDecomposition& decomp, const StepContext& ctx, int gx, int gy) {
    if (!decomp.owns_global(gx, gy)) {
        return;
    }

    const int lx = decomp.local_x_from_global(gx);
    const int ly = decomp.local_y_from_global(gy);

    const double v1_left = std::max(0.0, ctx.cur_v1[decomp.idx(lx - 1, ly)]);
    const double v1_right = std::max(0.0, ctx.cur_v1[decomp.idx(lx + 1, ly)]);
    const double v1_up = std::max(0.0, ctx.cur_v1[decomp.idx(lx, ly - 1)]);
    const double v1_down = std::max(0.0, ctx.cur_v1[decomp.idx(lx, ly + 1)]);

    const double v2_left = std::max(0.0, ctx.cur_v2[decomp.idx(lx - 1, ly)]);
    const double v2_right = std::max(0.0, ctx.cur_v2[decomp.idx(lx + 1, ly)]);
    const double v2_up = std::max(0.0, ctx.cur_v2[decomp.idx(lx, ly - 1)]);
    const double v2_down = std::max(0.0, ctx.cur_v2[decomp.idx(lx, ly + 1)]);

    const double max_v1 = std::max({v1_left, v1_right, v1_up, v1_down});
    const double max_v2 = std::max({v2_left, v2_right, v2_up, v2_down});

    const double avg_v1 = 0.25 * (v1_left + v1_right + v1_up + v1_down);
    const double avg_v2 = 0.25 * (v2_left + v2_right + v2_up + v2_down);

    ctx.next_v1[decomp.idx(lx, ly)] = ctx.alpha * max_v1 + (1.0 - ctx.alpha) * avg_v1;
    ctx.next_v2[decomp.idx(lx, ly)] = ctx.alpha * max_v2 + (1.0 - ctx.alpha) * avg_v2;
}

// Advances one local ant until local budget is consumed or the ant migrates out.
int process_local_ant(const DomainDecomposition& decomp, const StepContext& ctx, int& ant_x, int& ant_y, std::uint8_t& ant_loaded, std::uint64_t& ant_seed, std::size_t& local_food_counter, int my_rank, double& consumed_out) {
    consumed_out = 0.0;

    while (consumed_out < 1.0) {
        const int channel = (ant_loaded != 0) ? 1 : 0;
        const int old_x = ant_x;
        const int old_y = ant_y;

        const double p_left = (channel == 0)
            ? read_pheromone(decomp, ctx.cur_v1, old_x - 1, old_y)
            : read_pheromone(decomp, ctx.cur_v2, old_x - 1, old_y);
        const double p_right = (channel == 0)
            ? read_pheromone(decomp, ctx.cur_v1, old_x + 1, old_y)
            : read_pheromone(decomp, ctx.cur_v2, old_x + 1, old_y);
        const double p_up = (channel == 0)
            ? read_pheromone(decomp, ctx.cur_v1, old_x, old_y - 1)
            : read_pheromone(decomp, ctx.cur_v2, old_x, old_y - 1);
        const double p_down = (channel == 0)
            ? read_pheromone(decomp, ctx.cur_v1, old_x, old_y + 1)
            : read_pheromone(decomp, ctx.cur_v2, old_x, old_y + 1);

        const double max_phen = std::max({p_left, p_right, p_up, p_down});
        const double choice = rand_double(0.0, 1.0, ant_seed);

        int new_x = old_x;
        int new_y = old_y;

        if (choice > ctx.eps || max_phen <= 0.0) {
            while (true) {
                new_x = old_x;
                new_y = old_y;
                const int d = rand_int32(1, 4, ant_seed);
                if (d == 1) {
                    new_x -= 1;
                } else if (d == 2) {
                    new_y -= 1;
                } else if (d == 3) {
                    new_x += 1;
                } else {
                    new_y += 1;
                }

                const double p_candidate = (channel == 0)
                    ? read_pheromone(decomp, ctx.cur_v1, new_x, new_y)
                    : read_pheromone(decomp, ctx.cur_v2, new_x, new_y);
                if (p_candidate != -1.0) {
                    break;
                }
            }
        } else {
            if (p_left == max_phen) {
                new_x -= 1;
            } else if (p_right == max_phen) {
                new_x += 1;
            } else if (p_up == max_phen) {
                new_y -= 1;
            } else {
                new_y += 1;
            }
        }

        consumed_out += read_terrain(decomp, ctx.terrain, new_x, new_y);
        ant_x = new_x;
        ant_y = new_y;

        if (ant_x == ctx.pos_nest.x && ant_y == ctx.pos_nest.y) {
            if (ant_loaded != 0) {
                ++local_food_counter;
            }
            ant_loaded = 0;
        }

        if (ant_x == ctx.pos_food.x && ant_y == ctx.pos_food.y) {
            ant_loaded = 1;
        }

        const int owner = decomp.owner_of(ant_x, ant_y);
        if (owner == my_rank) {
            mark_cell_from_current(decomp, ctx, ant_x, ant_y);
        } else {
            return owner;
        }
    }

    return my_rank;
}

// Continues one migrant ant from its partial state until completion or remigration.
int process_single_ant(const DomainDecomposition& decomp, const StepContext& ctx, TransitAnt& ant, std::size_t& local_food_counter, int my_rank) {
    if (ant.pending_deposit != 0 && decomp.owns_global(ant.x, ant.y)) {
        mark_cell_from_current(decomp, ctx, ant.x, ant.y);
        ant.pending_deposit = 0;
    }

    while (ant.consumed < 1.0) {
        const int channel = (ant.loaded != 0) ? 1 : 0;
        const int old_x = ant.x;
        const int old_y = ant.y;

        const double p_left = (channel == 0)
            ? read_pheromone(decomp, ctx.cur_v1, old_x - 1, old_y)
            : read_pheromone(decomp, ctx.cur_v2, old_x - 1, old_y);
        const double p_right = (channel == 0)
            ? read_pheromone(decomp, ctx.cur_v1, old_x + 1, old_y)
            : read_pheromone(decomp, ctx.cur_v2, old_x + 1, old_y);
        const double p_up = (channel == 0)
            ? read_pheromone(decomp, ctx.cur_v1, old_x, old_y - 1)
            : read_pheromone(decomp, ctx.cur_v2, old_x, old_y - 1);
        const double p_down = (channel == 0)
            ? read_pheromone(decomp, ctx.cur_v1, old_x, old_y + 1)
            : read_pheromone(decomp, ctx.cur_v2, old_x, old_y + 1);

        const double max_phen = std::max({p_left, p_right, p_up, p_down});
        const double choice = rand_double(0.0, 1.0, ant.seed);

        int new_x = old_x;
        int new_y = old_y;

        if (choice > ctx.eps || max_phen <= 0.0) {
            // Random exploration until a non-blocked destination is found.
            while (true) {
                new_x = old_x;
                new_y = old_y;
                const int d = rand_int32(1, 4, ant.seed);
                if (d == 1) {
                    new_x -= 1;
                } else if (d == 2) {
                    new_y -= 1;
                } else if (d == 3) {
                    new_x += 1;
                } else {
                    new_y += 1;
                }

                const double p_candidate = (channel == 0)
                    ? read_pheromone(decomp, ctx.cur_v1, new_x, new_y)
                    : read_pheromone(decomp, ctx.cur_v2, new_x, new_y);
                if (p_candidate != -1.0) {
                    break;
                }
            }
        } else {
            if (p_left == max_phen) {
                new_x -= 1;
            } else if (p_right == max_phen) {
                new_x += 1;
            } else if (p_up == max_phen) {
                new_y -= 1;
            } else {
                new_y += 1;
            }
        }

        ant.consumed += read_terrain(decomp, ctx.terrain, new_x, new_y);
        ant.x = new_x;
        ant.y = new_y;

        if (ant.x == ctx.pos_nest.x && ant.y == ctx.pos_nest.y) {
            if (ant.loaded != 0) {
                ++local_food_counter;
            }
            ant.loaded = 0;
        }

        if (ant.x == ctx.pos_food.x && ant.y == ctx.pos_food.y) {
            ant.loaded = 1;
        }

        const int owner = decomp.owner_of(ant.x, ant.y);
        if (owner == my_rank) {
            mark_cell_from_current(decomp, ctx, ant.x, ant.y);
        } else {
            ant.pending_deposit = 1;
            return owner;
        }
    }

    return my_rank;
}

struct NeighborQueues {
    std::vector<TransitAnt> up;
    std::vector<TransitAnt> down;
    std::vector<TransitAnt> left;
    std::vector<TransitAnt> right;
};

void clear_neighbor_queues(NeighborQueues& queues) {
    queues.up.clear();
    queues.down.clear();
    queues.left.clear();
    queues.right.clear();
}

void reserve_neighbor_queues(NeighborQueues& queues, std::size_t cap) {
    queues.up.reserve(cap);
    queues.down.reserve(cap);
    queues.left.reserve(cap);
    queues.right.reserve(cap);
}

void route_ant_packet(const DomainDecomposition& decomp, const TransitAnt& ant, int owner, NeighborQueues& out) {
    if (owner == decomp.up_rank) {
        out.up.push_back(ant);
    } else if (owner == decomp.down_rank) {
        out.down.push_back(ant);
    } else if (owner == decomp.left_rank) {
        out.left.push_back(ant);
    } else if (owner == decomp.right_rank) {
        out.right.push_back(ant);
    } else if (owner != MPI_PROC_NULL) {
        if (ant.x < decomp.offset_x) {
            out.left.push_back(ant);
        } else if (ant.x >= decomp.offset_x + decomp.local_nx) {
            out.right.push_back(ant);
        } else if (ant.y < decomp.offset_y) {
            out.up.push_back(ant);
        } else {
            out.down.push_back(ant);
        }
    }
}

void exchange_migrants_with_neighbors(const DomainDecomposition& decomp, NeighborQueues& send,
                                      std::vector<TransitAnt>& received, std::vector<TransitAnt>& send_flat,
                                      std::vector<TransitAnt>& recv_flat, MPI_Comm comm, double& time_acc) {
    (void)decomp;
    const double t_mig_begin = MPI_Wtime();

    std::array<int, dir_count> send_counts{};
    send_counts[dir_up] = static_cast<int>(send.up.size());
    send_counts[dir_down] = static_cast<int>(send.down.size());
    send_counts[dir_left] = static_cast<int>(send.left.size());
    send_counts[dir_right] = static_cast<int>(send.right.size());

    std::array<int, dir_count> recv_counts{};
    MPI_Neighbor_alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    const std::size_t total_send = static_cast<std::size_t>(send_counts[dir_up] + send_counts[dir_down] +
                                                            send_counts[dir_left] + send_counts[dir_right]);
    send_flat.clear();
    send_flat.reserve(total_send);
    send_flat.insert(send_flat.end(), send.up.begin(), send.up.end());
    send_flat.insert(send_flat.end(), send.down.begin(), send.down.end());
    send_flat.insert(send_flat.end(), send.left.begin(), send.left.end());
    send_flat.insert(send_flat.end(), send.right.begin(), send.right.end());

    std::array<int, dir_count> send_counts_bytes{};
    std::array<int, dir_count> recv_counts_bytes{};
    std::array<int, dir_count> send_displs_bytes{};
    std::array<int, dir_count> recv_displs_bytes{};

    for (int i = 0; i < dir_count; ++i) {
        send_counts_bytes[i] = send_counts[i] * static_cast<int>(sizeof(TransitAnt));
        recv_counts_bytes[i] = recv_counts[i] * static_cast<int>(sizeof(TransitAnt));
    }

    for (int i = 1; i < dir_count; ++i) {
        send_displs_bytes[i] = send_displs_bytes[i - 1] + send_counts_bytes[i - 1];
        recv_displs_bytes[i] = recv_displs_bytes[i - 1] + recv_counts_bytes[i - 1];
    }

    const std::size_t total_recv =
        static_cast<std::size_t>(recv_counts[dir_up] + recv_counts[dir_down] + recv_counts[dir_left] + recv_counts[dir_right]);
    recv_flat.resize(total_recv);

    MPI_Neighbor_alltoallv(send_flat.empty() ? nullptr : send_flat.data(), send_counts_bytes.data(),
                           send_displs_bytes.data(), MPI_BYTE, recv_flat.empty() ? nullptr : recv_flat.data(),
                           recv_counts_bytes.data(), recv_displs_bytes.data(), MPI_BYTE, comm);

    received.swap(recv_flat);
    time_acc += MPI_Wtime() - t_mig_begin;
}

}  // namespace

// Generates ants on rank 0 and scatters them to owner ranks.
void distribute_initial_ants(const DomainDecomposition& decomp, int nb_ants, std::size_t global_seed, Population& local_ants, MPI_Comm comm) {
    local_ants.clear();

    std::vector<int> send_counts;
    std::vector<TransitAnt> flat;

    if (decomp.rank == 0) {
        send_counts.assign(decomp.size, 0);
        std::vector<std::vector<TransitAnt>> per_rank(static_cast<std::size_t>(decomp.size));

        std::size_t seed = global_seed;
        for (int i = 0; i < nb_ants; ++i) {
            const int gx = rand_int32(0, decomp.global_nx - 1, seed);
            const int gy = rand_int32(0, decomp.global_ny - 1, seed);
            const int owner = decomp.owner_of(gx, gy);
            TransitAnt ant{gx, gy, seed, 0.0, 0, 0};
            per_rank[static_cast<std::size_t>(owner)].push_back(ant);
        }

        flat.reserve(static_cast<std::size_t>(nb_ants));
        for (int r = 0; r < decomp.size; ++r) {
            send_counts[static_cast<std::size_t>(r)] = static_cast<int>(per_rank[static_cast<std::size_t>(r)].size());
            flat.insert(flat.end(), per_rank[static_cast<std::size_t>(r)].begin(), per_rank[static_cast<std::size_t>(r)].end());
        }
    }

    int local_count = 0;
    MPI_Scatter(send_counts.data(), 1, MPI_INT, &local_count, 1, MPI_INT, 0, comm);

    std::vector<int> send_counts_bytes;
    std::vector<int> displs_bytes;
    if (decomp.rank == 0) {
        send_counts_bytes.assign(decomp.size, 0);
        displs_bytes.assign(decomp.size, 0);
        int byte_displ = 0;
        for (int r = 0; r < decomp.size; ++r) {
            send_counts_bytes[static_cast<std::size_t>(r)] = send_counts[static_cast<std::size_t>(r)] * static_cast<int>(sizeof(TransitAnt));
            displs_bytes[static_cast<std::size_t>(r)] = byte_displ;
            byte_displ += send_counts_bytes[static_cast<std::size_t>(r)];
        }
    }

    std::vector<TransitAnt> local_packets(static_cast<std::size_t>(local_count));
    MPI_Scatterv(flat.data(),
                 send_counts_bytes.data(),
                 displs_bytes.data(),
                 MPI_BYTE,
                 local_packets.data(),
                 local_count * static_cast<int>(sizeof(TransitAnt)),
                 MPI_BYTE,
                 0,
                 comm);

    local_ants.reserve(static_cast<std::size_t>(local_count));
    for (const TransitAnt& ant : local_packets) {
        local_ants.add_ant(position_t{ant.x, ant.y}, static_cast<std::size_t>(ant.seed), ant.loaded);
    }
}

// Advances all local ants and handles migration rounds until convergence.
StepResult advance_ants_with_migration(const DomainDecomposition& decomp, const StepContext& ctx, Population& ants, MPI_Comm comm) {
    StepResult result;

    Population finished;
    finished.reserve(ants.size());

    // Reuse migration buffers across iterations to avoid repeated allocations.
    static NeighborQueues outgoing;
    static NeighborQueues next_outgoing;
    static std::vector<TransitAnt> active;
    static std::vector<TransitAnt> send_flat;
    static std::vector<TransitAnt> recv_flat;

    clear_neighbor_queues(outgoing);
    clear_neighbor_queues(next_outgoing);
    active.clear();
    send_flat.clear();
    recv_flat.clear();

    const std::size_t reserve_cap = ants.size() / 8 + 8;
    reserve_neighbor_queues(outgoing, reserve_cap);
    reserve_neighbor_queues(next_outgoing, reserve_cap);
    active.reserve(std::max(active.capacity(), reserve_cap));
    send_flat.reserve(std::max(send_flat.capacity(), reserve_cap));
    recv_flat.reserve(std::max(recv_flat.capacity(), reserve_cap));

    double t_move_begin = MPI_Wtime();
    for (std::size_t i = 0; i < ants.size(); ++i) {
        int ant_x = ants.pos_x(i);
        int ant_y = ants.pos_y(i);
        std::uint8_t ant_loaded = ants.state_at(i);
        std::uint64_t ant_seed = static_cast<std::uint64_t>(ants.seed_at(i));
        double consumed = 0.0;

        const int owner = process_local_ant(
            decomp,
            ctx,
            ant_x,
            ant_y,
            ant_loaded,
            ant_seed,
            result.food_collected_local,
            decomp.rank,
            consumed
        );

        if (owner == decomp.rank) {
            finished.add_ant(position_t{ant_x, ant_y}, static_cast<std::size_t>(ant_seed), ant_loaded);
        } else if (owner != MPI_PROC_NULL) {
            TransitAnt migrant{ant_x, ant_y, ant_seed, consumed, ant_loaded, 1};
            route_ant_packet(decomp, migrant, owner, outgoing);
        }
    }
    result.move_local_time += MPI_Wtime() - t_move_begin;

    exchange_migrants_with_neighbors(decomp, outgoing, active, send_flat, recv_flat, comm, result.migration_time);

    int local_active = static_cast<int>(active.size());
    int global_active = 0;
    MPI_Request active_req = MPI_REQUEST_NULL;
    MPI_Iallreduce(&local_active, &global_active, 1, MPI_INT, MPI_SUM, comm, &active_req);
    MPI_Wait(&active_req, MPI_STATUS_IGNORE);
    constexpr int migration_global_check_period = 2;
    int rounds_until_check = migration_global_check_period;

    while (global_active > 0) {
        t_move_begin = MPI_Wtime();
        clear_neighbor_queues(next_outgoing);
        const std::size_t needed_cap = active.size() / 8 + 8;
        reserve_neighbor_queues(next_outgoing, needed_cap);

        for (TransitAnt& ant : active) {
            const int owner = process_single_ant(decomp, ctx, ant, result.food_collected_local, decomp.rank);
            if (owner == decomp.rank) {
                finished.add_ant(position_t{ant.x, ant.y}, static_cast<std::size_t>(ant.seed), ant.loaded);
            } else {
                route_ant_packet(decomp, ant, owner, next_outgoing);
            }
        }

        result.move_local_time += MPI_Wtime() - t_move_begin;

        active.clear();
        exchange_migrants_with_neighbors(decomp, next_outgoing, active, send_flat, recv_flat, comm,
                                         result.migration_time);

        local_active = static_cast<int>(active.size());
        --rounds_until_check;
        if (rounds_until_check == 0) {
            MPI_Iallreduce(&local_active, &global_active, 1, MPI_INT, MPI_SUM, comm, &active_req);
            MPI_Wait(&active_req, MPI_STATUS_IGNORE);
            rounds_until_check = migration_global_check_period;
        } else {
            global_active = 1;
        }
    }

    ants = std::move(finished);
    return result;
}

}
