#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include "population.hpp"
#include "fractal_land.hpp"
#include "pheronome.hpp"
#include "rand_generator.hpp"
#include "renderer.hpp"
#include "window.hpp"

struct cli_options {
    bool headless = false;
    std::size_t max_iterations = 0;
    std::size_t warmup_iterations = 0;
    std::string timing_csv_path = "results/iter.csv";
    std::string summary_csv_path = "results/summary.csv";
};

struct advance_timing {
    double ants_ms = 0.0;
    double evaporation_ms = 0.0;
    double update_ms = 0.0;
    double total_ms = 0.0;
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

static double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                         const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static void print_usage(const char* progname) {
    std::cout
        << "Usage: " << progname << " [options]\n"
        << "Options:\n"
        << "  --headless                    Disable SDL rendering/event loop.\n"
        << "  --max-iterations N            Stop automatically after N iterations (0 = infinite).\n"
        << "  --warmup-iterations N         Iterations to skip in timing stats (default: 0).\n"
        << "  --timing-csv PATH             Write per-iteration timings to PATH (default: results/iter.csv).\n"
        << "  --summary-csv PATH            Write aggregated timing stats to PATH (default: results/summary.csv).\n"
        << "  --help                        Show this help message.\n";
}

static bool parse_size_t(const std::string& value, std::size_t& out) {
    if (value.empty()) return false;
    std::size_t pos = 0;
    try {
        unsigned long long parsed = std::stoull(value, &pos, 10);
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
        } else if (arg == "--headless") {
            opts.headless = true;
        } else if (arg == "--max-iterations") {
            if (i + 1 >= argc || !parse_size_t(argv[++i], opts.max_iterations)) {
                std::cerr << "Invalid value for --max-iterations\n";
                return false;
            }
        } else if (arg == "--warmup-iterations") {
            if (i + 1 >= argc || !parse_size_t(argv[++i], opts.warmup_iterations)) {
                std::cerr << "Invalid value for --warmup-iterations\n";
                return false;
            }
        } else if (arg == "--timing-csv") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --timing-csv\n";
                return false;
            }
            opts.timing_csv_path = argv[++i];
        } else if (arg == "--summary-csv") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --summary-csv\n";
                return false;
            }
            opts.summary_csv_path = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
    }

    if (opts.max_iterations > 0 && opts.warmup_iterations >= opts.max_iterations) {
        std::cerr << "--warmup-iterations must be smaller than --max-iterations\n";
        return false;
    }
    return true;
}

static advance_timing advance_time(const fractal_land& land, pheronome& phen, const position_t& pos_nest,
                                   const position_t& pos_food, Population& ants, std::size_t& food_counter) {
    const auto t0 = std::chrono::steady_clock::now();
    ants.advance_all(phen, land, pos_food, pos_nest, food_counter);
    const auto t1 = std::chrono::steady_clock::now();

    phen.do_evaporation();
    const auto t2 = std::chrono::steady_clock::now();

    phen.update();
    const auto t3 = std::chrono::steady_clock::now();

    advance_timing timings;
    timings.ants_ms = elapsed_ms(t0, t1);
    timings.evaporation_ms = elapsed_ms(t1, t2);
    timings.update_ms = elapsed_ms(t2, t3);
    timings.total_ms = elapsed_ms(t0, t3);
    return timings;
}

static void write_summary_metric(std::ostream& out, const std::string& metric_name, const running_stats& stats) {
    out << metric_name << "," << stats.count << "," << stats.sum << "," << stats.mean() << "," << stats.min << ","
        << stats.max << "\n";
}

int main(int argc, char* argv[]) {
    cli_options opts;
    if (!parse_cli(argc, argv, opts)) {
        print_usage(argv[0]);
        return 1;
    }

    std::size_t seed = 2026;
    const int nb_ants = 5000;
    const double eps = 0.8;
    const double alpha = 0.7;
    const double beta = 0.999;

    position_t pos_nest{256, 256};
    position_t pos_food{500, 500};

    fractal_land land(8, 2, 1., 1024);
    double max_val = 0.0;
    double min_val = 0.0;
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i) {
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            max_val = std::max(max_val, land(i, j));
            min_val = std::min(min_val, land(i, j));
        }
    }

    const double delta = max_val - min_val;
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i) {
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            land(i, j) = (land(i, j) - min_val) / delta;
        }
    }

    Population::set_exploration_coef(eps);

    Population ants;
    ants.reserve(static_cast<std::size_t>(nb_ants));
    auto gen_ant_pos = [&land, &seed]() { return rand_int32(0, static_cast<int>(land.dimensions() - 1), seed); };
    for (std::size_t i = 0; i < static_cast<std::size_t>(nb_ants); ++i) {
        ants.add_ant(position_t{gen_ant_pos(), gen_ant_pos()}, seed);
    }

    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    std::unique_ptr<Window> win;
    std::unique_ptr<Renderer> renderer;
    SDL_Event event;
    if (!opts.headless) {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
            return 1;
        }
        win = std::make_unique<Window>("Ant Simulation - Vectorized", 2 * land.dimensions() + 10,
                                       land.dimensions() + 266);
        renderer = std::make_unique<Renderer>(land, phen, pos_nest, pos_food, ants);
    }

    std::ofstream timing_csv;
    if (!opts.timing_csv_path.empty()) {
        const std::filesystem::path timing_path(opts.timing_csv_path);
        const std::filesystem::path timing_parent = timing_path.parent_path();
        if (!timing_parent.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(timing_parent, ec);
            if (ec) {
                std::cerr << "Cannot create directory for timing CSV: " << timing_parent << "\n";
                if (!opts.headless) SDL_Quit();
                return 1;
            }
        }

        timing_csv.open(opts.timing_csv_path);
        if (!timing_csv) {
            std::cerr << "Cannot open timing CSV: " << opts.timing_csv_path << "\n";
            if (!opts.headless) SDL_Quit();
            return 1;
        }

        timing_csv << std::fixed << std::setprecision(6);
        timing_csv << "iteration,food_quantity,event_poll_ms,ants_advance_ms,evaporation_ms,update_ms,advance_total_ms,render_ms,blit_ms,iteration_total_ms\n";
    }

    running_stats event_stats;
    running_stats ants_stats;
    running_stats evaporation_stats;
    running_stats update_stats;
    running_stats advance_total_stats;
    running_stats render_stats;
    running_stats blit_stats;
    running_stats iteration_total_stats;

    std::size_t food_quantity = 0;
    bool cont_loop = true;
    std::size_t first_food_iteration = 0;
    std::size_t it = 0;
    std::size_t measured_iterations = 0;

    while (cont_loop && (opts.max_iterations == 0 || it < opts.max_iterations)) {
        ++it;
        const auto iter_begin = std::chrono::steady_clock::now();

        double event_poll_ms = 0.0;
        if (!opts.headless) {
            const auto events_begin = std::chrono::steady_clock::now();
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) cont_loop = false;
            }
            const auto events_end = std::chrono::steady_clock::now();
            event_poll_ms = elapsed_ms(events_begin, events_end);
        }

        const advance_timing sim_timing = advance_time(land, phen, pos_nest, pos_food, ants, food_quantity);

        double render_ms = 0.0;
        double blit_ms = 0.0;
        if (!opts.headless) {
            const auto render_begin = std::chrono::steady_clock::now();
            renderer->display(*win, food_quantity);
            const auto render_end = std::chrono::steady_clock::now();
            win->blit();
            const auto blit_end = std::chrono::steady_clock::now();
            render_ms = elapsed_ms(render_begin, render_end);
            blit_ms = elapsed_ms(render_end, blit_end);
        }

        const auto iter_end = std::chrono::steady_clock::now();
        const double iteration_total_ms = elapsed_ms(iter_begin, iter_end);

        if (first_food_iteration == 0 && food_quantity > 0) {
            first_food_iteration = it;
            std::cout << "First food reached the nest at iteration " << it << "\n";
        }

        if (it > opts.warmup_iterations) {
            ++measured_iterations;
            event_stats.add(event_poll_ms);
            ants_stats.add(sim_timing.ants_ms);
            evaporation_stats.add(sim_timing.evaporation_ms);
            update_stats.add(sim_timing.update_ms);
            advance_total_stats.add(sim_timing.total_ms);
            render_stats.add(render_ms);
            blit_stats.add(blit_ms);
            iteration_total_stats.add(iteration_total_ms);

            if (timing_csv) {
                timing_csv << it << "," << food_quantity << "," << event_poll_ms << "," << sim_timing.ants_ms << ","
                           << sim_timing.evaporation_ms << "," << sim_timing.update_ms << "," << sim_timing.total_ms << ","
                           << render_ms << "," << blit_ms << "," << iteration_total_ms << "\n";
                timing_csv.flush();
            }
        }
    }

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
    std::cout << "timing_meta,measured_iterations," << measured_iterations << "\n";
    std::cout << "timing_meta,final_food_quantity," << food_quantity << "\n";
    if (first_food_iteration > 0) {
        std::cout << "timing_meta,first_food_iteration," << first_food_iteration << "\n";
    } else {
        std::cout << "timing_meta,first_food_iteration,not_reached\n";
    }

    if (!opts.summary_csv_path.empty()) {
        const std::filesystem::path summary_path(opts.summary_csv_path);
        const std::filesystem::path summary_parent = summary_path.parent_path();
        if (!summary_parent.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(summary_parent, ec);
            if (ec) {
                std::cerr << "Cannot create directory for summary CSV: " << summary_parent << "\n";
                if (!opts.headless) SDL_Quit();
                return 1;
            }
        }

        std::ofstream summary_csv(opts.summary_csv_path);
        if (!summary_csv) {
            std::cerr << "Cannot open summary CSV: " << opts.summary_csv_path << "\n";
            if (!opts.headless) SDL_Quit();
            return 1;
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

        summary_csv << "\nmeta_key,meta_value\n";
        summary_csv << "total_iterations," << it << "\n";
        summary_csv << "measured_iterations," << measured_iterations << "\n";
        summary_csv << "final_food_quantity," << food_quantity << "\n";
        if (first_food_iteration > 0) {
            summary_csv << "first_food_iteration," << first_food_iteration << "\n";
        } else {
            summary_csv << "first_food_iteration,not_reached\n";
        }
    }

    if (!opts.headless) SDL_Quit();
    return 0;
}
