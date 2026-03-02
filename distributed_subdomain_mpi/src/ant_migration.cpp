#include "ant_migration.hpp"

#include <algorithm>
#include <array>

#include "rand_generator.hpp"

namespace mpi_subdomain {

namespace {

constexpr int TAG_MIGRATION_COUNT_TO_UP = 900;
constexpr int TAG_MIGRATION_COUNT_TO_DOWN = 901;
constexpr int TAG_MIGRATION_COUNT_TO_LEFT = 902;
constexpr int TAG_MIGRATION_COUNT_TO_RIGHT = 903;
constexpr int TAG_MIGRATION_DATA_TO_UP = 910;
constexpr int TAG_MIGRATION_DATA_TO_DOWN = 911;
constexpr int TAG_MIGRATION_DATA_TO_LEFT = 912;
constexpr int TAG_MIGRATION_DATA_TO_RIGHT = 913;

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

// Routes one migrant ant packet to the proper neighbor queue.
void route_ant_packet(const DomainDecomposition& decomp, const TransitAnt& ant, int owner, std::vector<TransitAnt>& out_up, std::vector<TransitAnt>& out_down, std::vector<TransitAnt>& out_left, std::vector<TransitAnt>& out_right) {
    if (owner == decomp.up_rank) {
        out_up.push_back(ant);
    } else if (owner == decomp.down_rank) {
        out_down.push_back(ant);
    } else if (owner == decomp.left_rank) {
        out_left.push_back(ant);
    } else if (owner == decomp.right_rank) {
        out_right.push_back(ant);
    } else if (owner != MPI_PROC_NULL) {
        // Fallback routing for degenerate partitions with empty ranks.
        if (ant.x < decomp.offset_x) {
            out_left.push_back(ant);
        } else if (ant.x >= decomp.offset_x + decomp.local_nx) {
            out_right.push_back(ant);
        } else if (ant.y < decomp.offset_y) {
            out_up.push_back(ant);
        } else {
            out_down.push_back(ant);
        }
    }
}

// Exchanges migrant counts and packets with direct neighbors.
void exchange_migrants_with_neighbors(const DomainDecomposition& decomp, std::vector<TransitAnt>& send_up, std::vector<TransitAnt>& send_down, std::vector<TransitAnt>& send_left, std::vector<TransitAnt>& send_right, std::vector<TransitAnt>& received, MPI_Comm comm, double& time_acc) {
    int recv_count_up = 0;
    int recv_count_down = 0;
    int recv_count_left = 0;
    int recv_count_right = 0;
    const int send_count_up = static_cast<int>(send_up.size());
    const int send_count_down = static_cast<int>(send_down.size());
    const int send_count_left = static_cast<int>(send_left.size());
    const int send_count_right = static_cast<int>(send_right.size());

    const double t_mig_begin = MPI_Wtime();

    std::array<MPI_Request, 8> req_count{};
    int req_count_n = 0;
    if (decomp.up_rank != MPI_PROC_NULL) {
        MPI_Irecv(&recv_count_up, 1, MPI_INT, decomp.up_rank, TAG_MIGRATION_COUNT_TO_DOWN, comm, &req_count[req_count_n++]);
        MPI_Isend(&send_count_up, 1, MPI_INT, decomp.up_rank, TAG_MIGRATION_COUNT_TO_UP, comm, &req_count[req_count_n++]);
    }
    if (decomp.down_rank != MPI_PROC_NULL) {
        MPI_Irecv(&recv_count_down, 1, MPI_INT, decomp.down_rank, TAG_MIGRATION_COUNT_TO_UP, comm, &req_count[req_count_n++]);
        MPI_Isend(&send_count_down, 1, MPI_INT, decomp.down_rank, TAG_MIGRATION_COUNT_TO_DOWN, comm, &req_count[req_count_n++]);
    }
    if (decomp.left_rank != MPI_PROC_NULL) {
        MPI_Irecv(&recv_count_left, 1, MPI_INT, decomp.left_rank, TAG_MIGRATION_COUNT_TO_RIGHT, comm, &req_count[req_count_n++]);
        MPI_Isend(&send_count_left, 1, MPI_INT, decomp.left_rank, TAG_MIGRATION_COUNT_TO_LEFT, comm, &req_count[req_count_n++]);
    }
    if (decomp.right_rank != MPI_PROC_NULL) {
        MPI_Irecv(&recv_count_right, 1, MPI_INT, decomp.right_rank, TAG_MIGRATION_COUNT_TO_LEFT, comm, &req_count[req_count_n++]);
        MPI_Isend(&send_count_right, 1, MPI_INT, decomp.right_rank, TAG_MIGRATION_COUNT_TO_RIGHT, comm, &req_count[req_count_n++]);
    }
    if (req_count_n > 0) {
        MPI_Waitall(req_count_n, req_count.data(), MPI_STATUSES_IGNORE);
    }

    std::vector<TransitAnt> recv_up(static_cast<std::size_t>(recv_count_up));
    std::vector<TransitAnt> recv_down(static_cast<std::size_t>(recv_count_down));
    std::vector<TransitAnt> recv_left(static_cast<std::size_t>(recv_count_left));
    std::vector<TransitAnt> recv_right(static_cast<std::size_t>(recv_count_right));

    std::array<MPI_Request, 8> req_data{};
    int req_data_n = 0;
    if (decomp.up_rank != MPI_PROC_NULL) {
        MPI_Irecv(recv_up.data(), recv_count_up * static_cast<int>(sizeof(TransitAnt)), MPI_BYTE,
                  decomp.up_rank, TAG_MIGRATION_DATA_TO_DOWN, comm, &req_data[req_data_n++]);
        MPI_Isend(send_up.data(), send_count_up * static_cast<int>(sizeof(TransitAnt)), MPI_BYTE,
                  decomp.up_rank, TAG_MIGRATION_DATA_TO_UP, comm, &req_data[req_data_n++]);
    }
    if (decomp.down_rank != MPI_PROC_NULL) {
        MPI_Irecv(recv_down.data(), recv_count_down * static_cast<int>(sizeof(TransitAnt)), MPI_BYTE,
                  decomp.down_rank, TAG_MIGRATION_DATA_TO_UP, comm, &req_data[req_data_n++]);
        MPI_Isend(send_down.data(), send_count_down * static_cast<int>(sizeof(TransitAnt)), MPI_BYTE,
                  decomp.down_rank, TAG_MIGRATION_DATA_TO_DOWN, comm, &req_data[req_data_n++]);
    }
    if (decomp.left_rank != MPI_PROC_NULL) {
        MPI_Irecv(recv_left.data(), recv_count_left * static_cast<int>(sizeof(TransitAnt)), MPI_BYTE,
                  decomp.left_rank, TAG_MIGRATION_DATA_TO_RIGHT, comm, &req_data[req_data_n++]);
        MPI_Isend(send_left.data(), send_count_left * static_cast<int>(sizeof(TransitAnt)), MPI_BYTE,
                  decomp.left_rank, TAG_MIGRATION_DATA_TO_LEFT, comm, &req_data[req_data_n++]);
    }
    if (decomp.right_rank != MPI_PROC_NULL) {
        MPI_Irecv(recv_right.data(), recv_count_right * static_cast<int>(sizeof(TransitAnt)), MPI_BYTE,
                  decomp.right_rank, TAG_MIGRATION_DATA_TO_LEFT, comm, &req_data[req_data_n++]);
        MPI_Isend(send_right.data(), send_count_right * static_cast<int>(sizeof(TransitAnt)), MPI_BYTE,
                  decomp.right_rank, TAG_MIGRATION_DATA_TO_RIGHT, comm, &req_data[req_data_n++]);
    }
    if (req_data_n > 0) {
        MPI_Waitall(req_data_n, req_data.data(), MPI_STATUSES_IGNORE);
    }

    received.clear();
    received.reserve(static_cast<std::size_t>(recv_count_up + recv_count_down + recv_count_left + recv_count_right));
    received.insert(received.end(), recv_up.begin(), recv_up.end());
    received.insert(received.end(), recv_down.begin(), recv_down.end());
    received.insert(received.end(), recv_left.begin(), recv_left.end());
    received.insert(received.end(), recv_right.begin(), recv_right.end());

    time_acc += MPI_Wtime() - t_mig_begin;
}

}  // namespace

// Generates ants on rank 0 and scatters them to owner ranks.
void distribute_initial_ants(const DomainDecomposition& decomp, int nb_ants, std::size_t global_seed, AntSystem& local_ants, MPI_Comm comm) {
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
StepResult advance_ants_with_migration(const DomainDecomposition& decomp, const StepContext& ctx, AntSystem& ants, MPI_Comm comm) {
    StepResult result;

    AntSystem finished;
    finished.reserve(ants.size());

    std::vector<TransitAnt> out_up;
    std::vector<TransitAnt> out_down;
    std::vector<TransitAnt> out_left;
    std::vector<TransitAnt> out_right;
    out_up.reserve(ants.size() / 8 + 8);
    out_down.reserve(ants.size() / 8 + 8);
    out_left.reserve(ants.size() / 8 + 8);
    out_right.reserve(ants.size() / 8 + 8);

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
            route_ant_packet(decomp, migrant, owner, out_up, out_down, out_left, out_right);
        }
    }
    result.move_local_time += MPI_Wtime() - t_move_begin;

    std::vector<TransitAnt> active;
    exchange_migrants_with_neighbors(decomp, out_up, out_down, out_left, out_right, active, comm, result.migration_time);

    int local_active = static_cast<int>(active.size());
    int global_active = 0;
    MPI_Allreduce(&local_active, &global_active, 1, MPI_INT, MPI_SUM, comm);

    while (global_active > 0) {
        t_move_begin = MPI_Wtime();

        std::vector<TransitAnt> next_up;
        std::vector<TransitAnt> next_down;
        std::vector<TransitAnt> next_left;
        std::vector<TransitAnt> next_right;
        next_up.reserve(active.size() / 8 + 8);
        next_down.reserve(active.size() / 8 + 8);
        next_left.reserve(active.size() / 8 + 8);
        next_right.reserve(active.size() / 8 + 8);

        for (TransitAnt& ant : active) {
            const int owner = process_single_ant(decomp, ctx, ant, result.food_collected_local, decomp.rank);
            if (owner == decomp.rank) {
                finished.add_ant(position_t{ant.x, ant.y}, static_cast<std::size_t>(ant.seed), ant.loaded);
            } else {
                route_ant_packet(decomp, ant, owner, next_up, next_down, next_left, next_right);
            }
        }

        result.move_local_time += MPI_Wtime() - t_move_begin;

        active.clear();
        exchange_migrants_with_neighbors(decomp, next_up, next_down, next_left, next_right, active, comm, result.migration_time);

        local_active = static_cast<int>(active.size());
        MPI_Allreduce(&local_active, &global_active, 1, MPI_INT, MPI_SUM, comm);
    }

    ants = std::move(finished);
    return result;
}

}
