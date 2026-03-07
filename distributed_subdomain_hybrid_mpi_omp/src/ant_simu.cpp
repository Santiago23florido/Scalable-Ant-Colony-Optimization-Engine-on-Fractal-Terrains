#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../include/ant_migration.hpp"
#include "../include/population.hpp"
#include "../include/fractal_land.hpp"
#include "../include/halo.hpp"
#include "../include/pheronome.hpp"
#include "../include/renderer.hpp"
#include "../include/subdomain.hpp"
#include "../include/window.hpp"

struct cli_options {
    bool headless = false;
    std::size_t max_iterations = 0;
    std::size_t warmup_iterations = 0;
    std::size_t nb_ants = 0;
    std::size_t post_first_food_iterations = 0;
    std::string timing_csv_path = "results/iter.csv";
    std::string summary_csv_path = "results/summary.csv";
};

struct running_stats {
    std::size_t count = 0;
    double sum = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = 0.0;

    void add(double value) {
        ++count;
        sum += value;
        min = std::min(min, value);
        max = std::max(max, value);
    }

    double mean() const { return count == 0 ? 0.0 : sum / static_cast<double>(count); }
};

enum metric_idx {
    metric_ants_advance = 0,
    metric_evaporation = 1,
    metric_update = 2,
    metric_advance_total = 3,
    metric_mpi_halo = 4,
    metric_mpi_migration = 5,
    metric_mpi_food_allreduce = 6,
    metric_mpi_render_comm = 7,
    metric_mpi_total_comm = 8,
    metric_iteration_total = 9,
    metric_count = 10,
};

constexpr std::size_t metrics_reduce_batch_size = 16;

struct RenderAntPacket {
    int x;
    int y;
    std::uint8_t loaded;
    std::uint64_t seed;
};

volatile std::sig_atomic_t g_stop_requested = 0;

void on_termination_signal(int) { g_stop_requested = 1; }

static double elapsed_ms(const std::chrono::steady_clock::time_point& start, const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void print_usage(const char* progname) {
    std::cout
        << "Usage: " << progname << " [options]\n"
        << "Options:\n"
        << "  --headless                    Disable SDL rendering/event loop.\n"
        << "  --max-iterations N            Stop automatically after N iterations (0 = infinite).\n"
        << "  --warmup-iterations N         Iterations to skip in timing stats (default: 0).\n"
        << "  --nb-ants N                   Total ants globally (0 = default 1250 per MPI rank).\n"
        << "  --post-first-food-iterations N\n"
        << "                                Stop after N iterations once first food reaches the nest (0 = disabled).\n"
        << "  --timing-csv PATH             Write per-iteration timings to PATH (default: results/iter.csv).\n"
        << "  --summary-csv PATH            Write aggregated timing stats to PATH (default: results/summary.csv).\n"
        << "  --help                        Show this help message.\n";
}

static bool parse_size_t(const std::string& value, std::size_t& out) {
    if (value.empty()) return false;
    std::size_t pos = 0;
    try {
        const unsigned long long parsed = std::stoull(value, &pos, 10);
        if (pos != value.size()) return false;
        out = static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

static bool parse_cli(int argc, char* argv[], cli_options& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--headless") {
            opts.headless = true;
            continue;
        }
        if (arg == "--max-iterations") {
            if (i + 1 >= argc || !parse_size_t(argv[++i], opts.max_iterations)) {
                std::cerr << "Invalid value for --max-iterations\n";
                return false;
            }
            continue;
        }
        if (arg == "--warmup-iterations") {
            if (i + 1 >= argc || !parse_size_t(argv[++i], opts.warmup_iterations)) {
                std::cerr << "Invalid value for --warmup-iterations\n";
                return false;
            }
            continue;
        }
        if (arg == "--nb-ants") {
            if (i + 1 >= argc || !parse_size_t(argv[++i], opts.nb_ants)) {
                std::cerr << "Invalid value for --nb-ants\n";
                return false;
            }
            continue;
        }
        if (arg == "--post-first-food-iterations") {
            if (i + 1 >= argc || !parse_size_t(argv[++i], opts.post_first_food_iterations)) {
                std::cerr << "Invalid value for --post-first-food-iterations\n";
                return false;
            }
            continue;
        }
        if (arg == "--timing-csv") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --timing-csv\n";
                return false;
            }
            opts.timing_csv_path = argv[++i];
            continue;
        }
        if (arg == "--summary-csv") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --summary-csv\n";
                return false;
            }
            opts.summary_csv_path = argv[++i];
            continue;
        }

        std::cerr << "Unknown option: " << arg << "\n";
        return false;
    }

    if (opts.max_iterations > 0 && opts.warmup_iterations >= opts.max_iterations) {
        std::cerr << "--warmup-iterations must be smaller than --max-iterations\n";
        return false;
    }

    return true;
}

static void write_summary_metric(std::ostream& out, const std::string& metric_name, const running_stats& stats) {
    out << metric_name << "," << stats.count << "," << stats.sum << "," << stats.mean() << "," << stats.min << ","
        << stats.max << "\n";
}

static bool write_summary_csv_file(const std::string& summary_csv_path, const running_stats& event_stats, const running_stats& ants_stats, const running_stats& evaporation_stats, const running_stats& update_stats, const running_stats& advance_total_stats, const running_stats& render_stats, const running_stats& blit_stats, const running_stats& iteration_total_stats, const running_stats& mpi_halo_stats, const running_stats& mpi_migration_stats, const running_stats& mpi_food_allreduce_stats, const running_stats& mpi_render_comm_stats, const running_stats& mpi_total_comm_stats, std::size_t it, std::size_t measured_iterations, std::uint64_t food_quantity, std::size_t first_food_iteration) {
    if (summary_csv_path.empty()) {
        return true;
    }

    const std::filesystem::path summary_path(summary_csv_path);
    const std::filesystem::path summary_parent = summary_path.parent_path();
    if (!summary_parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(summary_parent, ec);
        if (ec) {
            std::cerr << "Cannot create directory for summary CSV: " << summary_parent << "\n";
            return false;
        }
    }

    std::ofstream summary_csv(summary_csv_path);
    if (!summary_csv) {
        std::cerr << "Cannot open summary CSV: " << summary_csv_path << "\n";
        return false;
    }

    summary_csv << std::fixed << std::setprecision(6);
    summary_csv << "metric,count,total_ms,mean_ms,min_ms,max_ms\n";
    write_summary_metric(summary_csv, "event_poll", event_stats);
    write_summary_metric(summary_csv, "ants_advance", ants_stats);
    write_summary_metric(summary_csv, "evaporation", evaporation_stats);
    write_summary_metric(summary_csv, "update", update_stats);
    write_summary_metric(summary_csv, "advance_total", advance_total_stats);
    write_summary_metric(summary_csv, "render", render_stats);
    write_summary_metric(summary_csv, "blit", blit_stats);
    write_summary_metric(summary_csv, "iteration_total", iteration_total_stats);
    write_summary_metric(summary_csv, "mpi_halo_exchange", mpi_halo_stats);
    write_summary_metric(summary_csv, "mpi_migration", mpi_migration_stats);
    write_summary_metric(summary_csv, "mpi_food_allreduce", mpi_food_allreduce_stats);
    write_summary_metric(summary_csv, "mpi_render_comm", mpi_render_comm_stats);
    write_summary_metric(summary_csv, "mpi_total_comm", mpi_total_comm_stats);

    summary_csv << "\nmeta_key,meta_value\n";
    summary_csv << "total_iterations," << it << "\n";
    summary_csv << "measured_iterations," << measured_iterations << "\n";
    summary_csv << "final_food_quantity," << food_quantity << "\n";
    if (first_food_iteration > 0) {
        summary_csv << "first_food_iteration," << first_food_iteration << "\n";
    } else {
        summary_csv << "first_food_iteration,not_reached\n";
    }

    summary_csv.flush();
    return true;
}

static int local_extent_for_coord(int global_n, int dims_n, int coord) {
    const int base = global_n / dims_n;
    const int rem = global_n % dims_n;
    return base + ((coord < rem) ? 1 : 0);
}

static int offset_for_coord(int global_n, int dims_n, int coord) {
    const int base = global_n / dims_n;
    const int rem = global_n % dims_n;
    return coord * base + std::min(coord, rem);
}

static void extract_local_interior(const mpi_subdomain::DomainDecomposition& decomp, const std::vector<double>& halo_field, std::vector<double>& dense_field) {
    dense_field.assign(static_cast<std::size_t>(decomp.local_nx * decomp.local_ny), 0.0);

    for (int ly = 0; ly < decomp.local_ny; ++ly) {
        for (int lx = 0; lx < decomp.local_nx; ++lx) {
            const std::size_t dense_idx = static_cast<std::size_t>(ly * decomp.local_nx + lx);
            dense_field[dense_idx] = halo_field[decomp.idx(lx + 1, ly + 1)];
        }
    }
}

static void unpack_rank_packed_field(const mpi_subdomain::DomainDecomposition& decomp, const std::vector<int>& displs, const std::vector<double>& packed_field, std::vector<double>& global_dense) {
    global_dense.assign(static_cast<std::size_t>(decomp.global_nx * decomp.global_ny), 0.0);

    for (int r = 0; r < decomp.size; ++r) {
        const std::size_t ridx = static_cast<std::size_t>(r);
        const int cx = decomp.rank_coord_x[ridx];
        const int cy = decomp.rank_coord_y[ridx];
        if (cx < 0 || cx >= decomp.dims_x || cy < 0 || cy >= decomp.dims_y) {
            continue;
        }

        const int nx = local_extent_for_coord(decomp.global_nx, decomp.dims_x, cx);
        const int ny = local_extent_for_coord(decomp.global_ny, decomp.dims_y, cy);
        const int ox = offset_for_coord(decomp.global_nx, decomp.dims_x, cx);
        const int oy = offset_for_coord(decomp.global_ny, decomp.dims_y, cy);
        const int disp = displs[ridx];

        for (int ly = 0; ly < ny; ++ly) {
            for (int lx = 0; lx < nx; ++lx) {
                const std::size_t dst = static_cast<std::size_t>((oy + ly) * decomp.global_nx + (ox + lx));
                const std::size_t src = static_cast<std::size_t>(disp + ly * nx + lx);
                global_dense[dst] = packed_field[src];
            }
        }
    }
}

static void gather_state_for_render(const mpi_subdomain::DomainDecomposition& decomp, const std::vector<int>& gather_counts, const std::vector<int>& gather_displs, const pheronome& local_phen, const Population& local_ants, pheronome* render_phen, Population* render_ants, MPI_Comm comm) {
    std::vector<double> local_dense_v1;
    std::vector<double> local_dense_v2;
    extract_local_interior(decomp, local_phen.current_channel(0), local_dense_v1);
    extract_local_interior(decomp, local_phen.current_channel(1), local_dense_v2);

    std::vector<double> packed_v1;
    std::vector<double> packed_v2;
    if (decomp.rank == 0) {
        packed_v1.resize(static_cast<std::size_t>(decomp.global_nx * decomp.global_ny));
        packed_v2.resize(static_cast<std::size_t>(decomp.global_nx * decomp.global_ny));
    }

    MPI_Gatherv(local_dense_v1.empty() ? nullptr : local_dense_v1.data(), static_cast<int>(local_dense_v1.size()), MPI_DOUBLE, (decomp.rank == 0) ? packed_v1.data() : nullptr, (decomp.rank == 0) ? gather_counts.data() : nullptr, (decomp.rank == 0) ? gather_displs.data() : nullptr, MPI_DOUBLE, 0, comm);

    MPI_Gatherv(local_dense_v2.empty() ? nullptr : local_dense_v2.data(), static_cast<int>(local_dense_v2.size()), MPI_DOUBLE, (decomp.rank == 0) ? packed_v2.data() : nullptr, (decomp.rank == 0) ? gather_counts.data() : nullptr, (decomp.rank == 0) ? gather_displs.data() : nullptr, MPI_DOUBLE, 0, comm);

    const int local_ant_count = static_cast<int>(local_ants.size());
    std::vector<int> ant_counts;
    if (decomp.rank == 0) {
        ant_counts.resize(static_cast<std::size_t>(decomp.size), 0);
    }

    MPI_Gather(&local_ant_count, 1, MPI_INT, (decomp.rank == 0) ? ant_counts.data() : nullptr, 1, MPI_INT, 0, comm);

    std::vector<RenderAntPacket> local_ant_packets(static_cast<std::size_t>(local_ant_count));
    for (int i = 0; i < local_ant_count; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        local_ant_packets[idx] = RenderAntPacket{
            local_ants.pos_x(idx),
            local_ants.pos_y(idx),
            local_ants.state_at(idx),
            static_cast<std::uint64_t>(local_ants.seed_at(idx)),
        };
    }

    std::vector<int> ant_counts_bytes;
    std::vector<int> ant_displs_bytes;
    std::vector<RenderAntPacket> gathered_ants;
    std::size_t total_ants = 0;

    if (decomp.rank == 0) {
        ant_counts_bytes.resize(static_cast<std::size_t>(decomp.size), 0);
        ant_displs_bytes.resize(static_cast<std::size_t>(decomp.size), 0);

        int byte_displ = 0;
        for (int r = 0; r < decomp.size; ++r) {
            const std::size_t ridx = static_cast<std::size_t>(r);
            ant_counts_bytes[ridx] = ant_counts[ridx] * static_cast<int>(sizeof(RenderAntPacket));
            ant_displs_bytes[ridx] = byte_displ;
            byte_displ += ant_counts_bytes[ridx];
            total_ants += static_cast<std::size_t>(ant_counts[ridx]);
        }

        gathered_ants.resize(total_ants);
    }

    MPI_Gatherv(local_ant_packets.empty() ? nullptr : local_ant_packets.data(), local_ant_count * static_cast<int>(sizeof(RenderAntPacket)), MPI_BYTE, (decomp.rank == 0) ? gathered_ants.data() : nullptr, (decomp.rank == 0) ? ant_counts_bytes.data() : nullptr, (decomp.rank == 0) ? ant_displs_bytes.data() : nullptr, MPI_BYTE, 0, comm);

    if (decomp.rank != 0) {
        return;
    }

    std::vector<double> global_dense_v1;
    std::vector<double> global_dense_v2;
    unpack_rank_packed_field(decomp, gather_displs, packed_v1, global_dense_v1);
    unpack_rank_packed_field(decomp, gather_displs, packed_v2, global_dense_v2);

    if (render_phen != nullptr) {
        render_phen->load_from_dense(global_dense_v1, global_dense_v2);
    }

    if (render_ants != nullptr) {
        render_ants->clear();
        render_ants->reserve(total_ants);
        for (const RenderAntPacket& packet : gathered_ants) {
            render_ants->add_ant(position_t{packet.x, packet.y}, static_cast<std::size_t>(packet.seed), packet.loaded);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    std::signal(SIGINT, on_termination_signal);
    std::signal(SIGTERM, on_termination_signal);

    MPI_Comm world_comm = MPI_COMM_WORLD;
    int world_rank = 0;
    MPI_Comm_rank(world_comm, &world_rank);

    cli_options opts;
    if (!parse_cli(argc, argv, opts)) {
        if (world_rank == 0) {
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const int ants_per_rank = 1250;
    const double eps = 0.8;
    const double alpha = 0.7;
    const double beta = 0.999;
    const std::size_t land_seed = 1024;
    const std::size_t ants_seed = 2026;

    const position_t pos_nest{256, 256};
    const position_t pos_food{500, 500};

    std::unique_ptr<fractal_land> global_land;
    int global_dim = 0;

    if (world_rank == 0) {
        global_land = std::make_unique<fractal_land>(8, 2, 1.0, static_cast<int>(land_seed));
        global_land->normalize_land();
        global_dim = static_cast<int>(global_land->dimensions());
    }

    MPI_Bcast(&global_dim, 1, MPI_INT, 0, world_comm);

    MPI_Comm comm = MPI_COMM_NULL;
    const auto decomp = mpi_subdomain::map_decomposed(global_dim, global_dim, world_comm, comm);
    const int rank = decomp.rank;
    const std::size_t default_nb_ants =
        static_cast<std::size_t>(ants_per_rank) * static_cast<std::size_t>(decomp.size);
    const std::size_t total_nb_ants = (opts.nb_ants > 0) ? opts.nb_ants : default_nb_ants;
    if (total_nb_ants > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::cerr << "nb_ants is too large for current MPI packet format\n";
        }
        if (comm != MPI_COMM_NULL) {
            MPI_Comm_free(&comm);
        }
        MPI_Finalize();
        return 1;
    }
    const int nb_ants = static_cast<int>(total_nb_ants);

    const std::vector<int> gather_counts = decomp.gather_counts_cells();
    const std::vector<int> gather_displs = decomp.gather_displs_cells();

    std::vector<double> packed_land;
    if (rank == 0) {
        packed_land = global_land->pack_land_by_rank(decomp, gather_displs);
    }

    std::vector<double> local_land_dense(static_cast<std::size_t>(decomp.local_nx * decomp.local_ny), 1.0);
    MPI_Scatterv((rank == 0) ? packed_land.data() : nullptr, gather_counts.data(), gather_displs.data(), MPI_DOUBLE, local_land_dense.empty() ? nullptr : local_land_dense.data(), static_cast<int>(local_land_dense.size()), MPI_DOUBLE, 0, comm);

    std::vector<double> terrain(decomp.halo_size(), 1.0);
    std::vector<int> cell_type(decomp.halo_size(), 0);
    for (int ly = 1; ly <= decomp.local_ny; ++ly) {
        for (int lx = 1; lx <= decomp.local_nx; ++lx) {
            const std::size_t dense_idx = static_cast<std::size_t>((ly - 1) * decomp.local_nx + (lx - 1));
            terrain[decomp.idx(lx, ly)] = local_land_dense[dense_idx];
            cell_type[decomp.idx(lx, ly)] = 0;
        }
    }

    if (decomp.owns_global(pos_nest.x, pos_nest.y)) {
        const int lx = decomp.local_x_from_global(pos_nest.x);
        const int ly = decomp.local_y_from_global(pos_nest.y);
        cell_type[decomp.idx(lx, ly)] = 1;
    }
    if (decomp.owns_global(pos_food.x, pos_food.y)) {
        const int lx = decomp.local_x_from_global(pos_food.x);
        const int ly = decomp.local_y_from_global(pos_food.y);
        cell_type[decomp.idx(lx, ly)] = 2;
    }

    mpi_subdomain::exchange_static_halos(decomp, terrain, cell_type, comm);

    pheronome local_phen(decomp, pos_food, pos_nest, alpha, beta);
    mpi_subdomain::PheromoneHaloExchange halo_exchange_state;
    mpi_subdomain::exchange_pheromone_halos(decomp, local_phen.current_channel(0), local_phen.current_channel(1), comm);

    Population::set_exploration_coef(eps);

    Population ants;
    mpi_subdomain::distribute_initial_ants(decomp, nb_ants, ants_seed, ants, comm);

    std::unique_ptr<Window> win;
    std::unique_ptr<pheronome> render_phen;
    std::unique_ptr<Population> render_ants;
    std::unique_ptr<Renderer> renderer;

    if (rank == 0 && !opts.headless) {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
            MPI_Abort(comm, 1);
        }

        win = std::make_unique<Window>("Ant Simulation - Hybrid MPI+OpenMP", 2 * global_dim + 10, global_dim + 266);
        render_phen = std::make_unique<pheronome>(static_cast<unsigned long>(global_dim), pos_food, pos_nest, alpha, beta);
        render_ants = std::make_unique<Population>();
        render_ants->reserve(total_nb_ants);
        renderer = std::make_unique<Renderer>(*global_land, *render_phen, pos_nest, pos_food, *render_ants);
    }

    std::ofstream timing_csv;
    if (rank == 0 && !opts.timing_csv_path.empty()) {
        const std::filesystem::path timing_path(opts.timing_csv_path);
        const std::filesystem::path timing_parent = timing_path.parent_path();
        if (!timing_parent.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(timing_parent, ec);
            if (ec) {
                std::cerr << "Cannot create directory for timing CSV: " << timing_parent << "\n";
                if (!opts.headless) SDL_Quit();
                MPI_Finalize();
                return 1;
            }
        }

        timing_csv.open(opts.timing_csv_path);
        if (!timing_csv) {
            std::cerr << "Cannot open timing CSV: " << opts.timing_csv_path << "\n";
            if (!opts.headless) SDL_Quit();
            MPI_Finalize();
            return 1;
        }

        timing_csv << std::fixed << std::setprecision(6);
        timing_csv
            << "iteration,food_quantity,event_poll_ms,ants_advance_ms,evaporation_ms,update_ms,advance_total_ms,render_ms,blit_ms,iteration_total_ms,mpi_halo_exchange_ms,mpi_migration_ms,mpi_food_allreduce_ms,mpi_render_comm_ms,mpi_total_comm_ms\n";
    }

    running_stats event_stats;
    running_stats ants_stats;
    running_stats evaporation_stats;
    running_stats update_stats;
    running_stats advance_total_stats;
    running_stats render_stats;
    running_stats blit_stats;
    running_stats iteration_total_stats;
    running_stats mpi_halo_stats;
    running_stats mpi_migration_stats;
    running_stats mpi_food_allreduce_stats;
    running_stats mpi_render_comm_stats;
    running_stats mpi_total_comm_stats;

    std::uint64_t food_quantity = 0;
    std::size_t first_food_iteration = 0;
    std::size_t it = 0;
    std::size_t measured_iterations = 0;
    bool keep_running = true;

    std::vector<double> local_metric_batch;
    local_metric_batch.reserve(metrics_reduce_batch_size * metric_count);
    std::vector<double> global_metric_batch;
    std::vector<double> rank0_event_poll_batch;
    std::vector<double> rank0_render_batch;
    std::vector<double> rank0_blit_batch;
    std::vector<std::uint64_t> rank0_food_batch;
    std::vector<std::size_t> rank0_iteration_batch;
    if (rank == 0) {
        rank0_event_poll_batch.reserve(metrics_reduce_batch_size);
        rank0_render_batch.reserve(metrics_reduce_batch_size);
        rank0_blit_batch.reserve(metrics_reduce_batch_size);
        rank0_food_batch.reserve(metrics_reduce_batch_size);
        rank0_iteration_batch.reserve(metrics_reduce_batch_size);
    }

    auto flush_metric_batch = [&](bool force_flush) {
        const std::size_t batch_count = local_metric_batch.size() / metric_count;
        if (batch_count == 0) return;
        if (!force_flush && batch_count < metrics_reduce_batch_size) return;

        if (rank == 0) {
            global_metric_batch.resize(batch_count * metric_count);
        }

        MPI_Reduce(local_metric_batch.data(), (rank == 0) ? global_metric_batch.data() : nullptr,
                   static_cast<int>(batch_count * metric_count), MPI_DOUBLE, MPI_MAX, 0, comm);

        if (rank == 0) {
            for (std::size_t i = 0; i < batch_count; ++i) {
                const std::size_t base = i * metric_count;
                event_stats.add(rank0_event_poll_batch[i]);
                ants_stats.add(global_metric_batch[base + metric_ants_advance]);
                evaporation_stats.add(global_metric_batch[base + metric_evaporation]);
                update_stats.add(global_metric_batch[base + metric_update]);
                advance_total_stats.add(global_metric_batch[base + metric_advance_total]);
                render_stats.add(rank0_render_batch[i]);
                blit_stats.add(rank0_blit_batch[i]);
                iteration_total_stats.add(global_metric_batch[base + metric_iteration_total]);
                mpi_halo_stats.add(global_metric_batch[base + metric_mpi_halo]);
                mpi_migration_stats.add(global_metric_batch[base + metric_mpi_migration]);
                mpi_food_allreduce_stats.add(global_metric_batch[base + metric_mpi_food_allreduce]);
                mpi_render_comm_stats.add(global_metric_batch[base + metric_mpi_render_comm]);
                mpi_total_comm_stats.add(global_metric_batch[base + metric_mpi_total_comm]);

                if (timing_csv) {
                    timing_csv << rank0_iteration_batch[i] << "," << rank0_food_batch[i] << ","
                               << rank0_event_poll_batch[i] << "," << global_metric_batch[base + metric_ants_advance]
                               << "," << global_metric_batch[base + metric_evaporation] << ","
                               << global_metric_batch[base + metric_update] << ","
                               << global_metric_batch[base + metric_advance_total] << ","
                               << rank0_render_batch[i] << "," << rank0_blit_batch[i] << ","
                               << global_metric_batch[base + metric_iteration_total] << ","
                               << global_metric_batch[base + metric_mpi_halo] << ","
                               << global_metric_batch[base + metric_mpi_migration] << ","
                               << global_metric_batch[base + metric_mpi_food_allreduce] << ","
                               << global_metric_batch[base + metric_mpi_render_comm] << ","
                               << global_metric_batch[base + metric_mpi_total_comm] << "\n";
                }
            }

            rank0_event_poll_batch.clear();
            rank0_render_batch.clear();
            rank0_blit_batch.clear();
            rank0_food_batch.clear();
            rank0_iteration_batch.clear();
        }

        local_metric_batch.clear();
    };

    while (true) {
        int local_stop = (g_stop_requested != 0) ? 1 : 0;
        int global_stop = 0;
        MPI_Allreduce(&local_stop, &global_stop, 1, MPI_INT, MPI_MAX, comm);
        if (global_stop != 0) {
            keep_running = false;
        }

        if (rank == 0 && opts.max_iterations > 0 && it >= opts.max_iterations) {
            keep_running = false;
        }
        if (rank == 0 && opts.post_first_food_iterations > 0 && first_food_iteration > 0 &&
            it >= first_food_iteration + opts.post_first_food_iterations) {
            keep_running = false;
        }

        int continue_flag = keep_running ? 1 : 0;
        MPI_Bcast(&continue_flag, 1, MPI_INT, 0, comm);
        if (continue_flag == 0) {
            break;
        }

        ++it;
        const auto iter_begin = std::chrono::steady_clock::now();

        double event_poll_ms = 0.0;
        if (rank == 0 && !opts.headless) {
            SDL_Event event;
            const auto events_begin = std::chrono::steady_clock::now();
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    keep_running = false;
                }
            }
            const auto events_end = std::chrono::steady_clock::now();
            event_poll_ms = elapsed_ms(events_begin, events_end);
        }

        double halo_ms_local = 0.0;
        {
            const double t0 = MPI_Wtime();
            mpi_subdomain::begin_pheromone_halo_exchange(decomp, local_phen.current_channel(0), local_phen.current_channel(1), halo_exchange_state, comm);
            local_phen.copy_current_to_buffer();
            mpi_subdomain::end_pheromone_halo_exchange(decomp, local_phen.current_channel(0), local_phen.current_channel(1), halo_exchange_state);
            halo_ms_local = (MPI_Wtime() - t0) * 1000.0;
        }

        mpi_subdomain::StepContext step_ctx{terrain, local_phen.current_channel(0), local_phen.current_channel(1), local_phen.buffer_channel(0), local_phen.buffer_channel(1), pos_food, pos_nest, alpha, eps};

        const auto step_result = mpi_subdomain::advance_ants_with_migration(decomp, step_ctx, ants, comm);
        const double ants_ms_local = step_result.move_local_time * 1000.0;
        const double migration_ms_local = step_result.migration_time * 1000.0;
        const std::uint64_t local_food_delta = static_cast<std::uint64_t>(step_result.food_collected_local);
        std::uint64_t global_food_delta = 0;
        double food_allreduce_ms_local = 0.0;
        MPI_Request food_allreduce_req = MPI_REQUEST_NULL;
        MPI_Iallreduce(&local_food_delta, &global_food_delta, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm,
                       &food_allreduce_req);

        double evaporation_ms_local = 0.0;
        {
            const double t0 = MPI_Wtime();
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
            for (int ly = 1; ly <= decomp.local_ny; ++ly) {
                for (int lx = 1; lx <= decomp.local_nx; ++lx) {
                    const std::size_t idx = decomp.idx(lx, ly);
                    local_phen.buffer_channel(0)[idx] *= beta;
                    local_phen.buffer_channel(1)[idx] *= beta;
                }
            }
            evaporation_ms_local = (MPI_Wtime() - t0) * 1000.0;
        }

        double update_ms_local = 0.0;
        {
            const double t0 = MPI_Wtime();
            if (decomp.owns_global(pos_food.x, pos_food.y)) {
                const int lx = decomp.local_x_from_global(pos_food.x);
                const int ly = decomp.local_y_from_global(pos_food.y);
                local_phen.buffer_channel(0)[decomp.idx(lx, ly)] = 1.0;
            }
            if (decomp.owns_global(pos_nest.x, pos_nest.y)) {
                const int lx = decomp.local_x_from_global(pos_nest.x);
                const int ly = decomp.local_y_from_global(pos_nest.y);
                local_phen.buffer_channel(1)[decomp.idx(lx, ly)] = 1.0;
            }

            local_phen.swap_current_with_buffer();
            local_phen.set_current_physical_ghosts(-1.0);
            update_ms_local = (MPI_Wtime() - t0) * 1000.0;
        }

        const double advance_total_local = ants_ms_local + evaporation_ms_local + update_ms_local;

        double render_comm_ms_local = 0.0;
        if (!opts.headless) {
            const double t0 = MPI_Wtime();
            gather_state_for_render(decomp, gather_counts, gather_displs, local_phen, ants, (rank == 0) ? render_phen.get() : nullptr, (rank == 0) ? render_ants.get() : nullptr, comm);
            render_comm_ms_local = (MPI_Wtime() - t0) * 1000.0;
        }

        {
            const double wait_t0 = MPI_Wtime();
            MPI_Wait(&food_allreduce_req, MPI_STATUS_IGNORE);
            food_allreduce_ms_local = (MPI_Wtime() - wait_t0) * 1000.0;
        }
        food_quantity += global_food_delta;

        double render_ms = 0.0;
        double blit_ms = 0.0;
        if (rank == 0 && !opts.headless) {
            const auto render_begin = std::chrono::steady_clock::now();
            renderer->display(*win, static_cast<std::size_t>(food_quantity));
            const auto render_end = std::chrono::steady_clock::now();
            win->blit();
            const auto blit_end = std::chrono::steady_clock::now();
            render_ms = elapsed_ms(render_begin, render_end);
            blit_ms = elapsed_ms(render_end, blit_end);
        }

        const double mpi_total_comm_local =
            halo_ms_local + migration_ms_local + food_allreduce_ms_local + render_comm_ms_local;

        const auto iter_end = std::chrono::steady_clock::now();
        const double iteration_total_ms_local = elapsed_ms(iter_begin, iter_end);

        double local_metrics[metric_count] = {
            ants_ms_local,
            evaporation_ms_local,
            update_ms_local,
            advance_total_local,
            halo_ms_local,
            migration_ms_local,
            food_allreduce_ms_local,
            render_comm_ms_local,
            mpi_total_comm_local,
            iteration_total_ms_local,
        };

        if (it > opts.warmup_iterations) {
            local_metric_batch.insert(local_metric_batch.end(), std::begin(local_metrics), std::end(local_metrics));
        }

        if (rank == 0) {
            if (first_food_iteration == 0 && food_quantity > 0) {
                first_food_iteration = it;
                std::cout << "First food reached the nest at iteration " << it << "\n";
            }

            if (it > opts.warmup_iterations) {
                ++measured_iterations;
                rank0_iteration_batch.push_back(it);
                rank0_food_batch.push_back(food_quantity);
                rank0_event_poll_batch.push_back(event_poll_ms);
                rank0_render_batch.push_back(render_ms);
                rank0_blit_batch.push_back(blit_ms);
            }
        }

        if (it > opts.warmup_iterations) {
            flush_metric_batch(false);
        }
    }

    flush_metric_batch(true);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "timing_summary,metric,count,total_ms,mean_ms,min_ms,max_ms\n";
        write_summary_metric(std::cout, "event_poll", event_stats);
        write_summary_metric(std::cout, "ants_advance", ants_stats);
        write_summary_metric(std::cout, "evaporation", evaporation_stats);
        write_summary_metric(std::cout, "update", update_stats);
        write_summary_metric(std::cout, "advance_total", advance_total_stats);
        write_summary_metric(std::cout, "render", render_stats);
        write_summary_metric(std::cout, "blit", blit_stats);
        write_summary_metric(std::cout, "iteration_total", iteration_total_stats);
        write_summary_metric(std::cout, "mpi_halo_exchange", mpi_halo_stats);
        write_summary_metric(std::cout, "mpi_migration", mpi_migration_stats);
        write_summary_metric(std::cout, "mpi_food_allreduce", mpi_food_allreduce_stats);
        write_summary_metric(std::cout, "mpi_render_comm", mpi_render_comm_stats);
        write_summary_metric(std::cout, "mpi_total_comm", mpi_total_comm_stats);
        std::cout << "timing_meta,measured_iterations," << measured_iterations << "\n";
        std::cout << "timing_meta,final_food_quantity," << food_quantity << "\n";
        if (first_food_iteration > 0) {
            std::cout << "timing_meta,first_food_iteration," << first_food_iteration << "\n";
        } else {
            std::cout << "timing_meta,first_food_iteration,not_reached\n";
        }

        write_summary_csv_file(opts.summary_csv_path, event_stats, ants_stats, evaporation_stats, update_stats, advance_total_stats, render_stats, blit_stats, iteration_total_stats, mpi_halo_stats, mpi_migration_stats, mpi_food_allreduce_stats, mpi_render_comm_stats, mpi_total_comm_stats, it, measured_iterations, food_quantity, first_food_iteration);
    }

    if (rank == 0 && !opts.headless) {
        SDL_Quit();
    }
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_free(&comm);
    }

    MPI_Finalize();
    return 0;
}
