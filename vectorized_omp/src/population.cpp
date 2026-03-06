#include "population.hpp"

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "rand_generator.hpp"

double Population::m_eps = 0.0;

void Population::reserve(std::size_t count) {
    m_pos_x.reserve(count);
    m_pos_y.reserve(count);
    m_state.reserve(count);
    m_seed.reserve(count);
}

void Population::add_ant(const position_t& pos, std::size_t seed) {
    m_pos_x.push_back(pos.x);
    m_pos_y.push_back(pos.y);
    m_state.push_back(unloaded);
    m_seed.push_back(seed);
}

void Population::advance_all(pheronome& phen, const fractal_land& land, const position_t& pos_food,
                            const position_t& pos_nest, std::size_t& food_counter) {
    const std::size_t ant_count = size();
    if (ant_count == 0) return;

    int thread_count = 1;
#ifdef _OPENMP
    thread_count = omp_get_max_threads();
#endif

    std::vector<std::vector<pheronome::size_t>> touched_by_thread(static_cast<std::size_t>(thread_count));
    std::size_t food_delta = 0;

#ifdef _OPENMP
#pragma omp parallel reduction(+ : food_delta)
    {
        const int tid = omp_get_thread_num();
        auto& local_touched = touched_by_thread[static_cast<std::size_t>(tid)];
        local_touched.clear();
        local_touched.reserve(ant_count / static_cast<std::size_t>(thread_count) + 64);

#pragma omp for schedule(static)
        for (std::size_t i = 0; i < ant_count; ++i) {
            std::size_t local_food_counter = 0;
            advance_one(i, phen, land, pos_food, pos_nest, local_food_counter, local_touched);
            food_delta += local_food_counter;
        }
    }
#else
    auto& local_touched = touched_by_thread[0];
    local_touched.reserve(ant_count + 64);
    for (std::size_t i = 0; i < ant_count; ++i) {
        std::size_t local_food_counter = 0;
        advance_one(i, phen, land, pos_food, pos_nest, local_food_counter, local_touched);
        food_delta += local_food_counter;
    }
#endif

    food_counter += food_delta;

    // All marks are derived from the same pheromone snapshot, so applying them here preserves behavior.
    for (const auto& local_touched : touched_by_thread) {
        for (const pheronome::size_t flat : local_touched) {
            phen.mark_pheronome_flat(flat);
        }
    }
}

void Population::advance_one(std::size_t idx, pheronome& phen, const fractal_land& land, const position_t& pos_food,
                            const position_t& pos_nest, std::size_t& local_food_counter,
                            std::vector<pheronome::size_t>& touched_cells) {
    int x = m_pos_x[idx];
    int y = m_pos_y[idx];
    std::uint8_t state = m_state[idx];
    std::size_t local_seed = m_seed[idx];
    const std::size_t stride = phen.stride();
    double consumed_time = 0.0;

    while (consumed_time < 1.0) {
        const int ind_pher = (state == loaded) ? 1 : 0;
        const double choice = rand_double(0.0, 1.0, local_seed);

        const int old_x = x;
        const int old_y = y;
        const std::size_t old_flat = phen.flat_index(old_x, old_y);

        int new_x = old_x;
        int new_y = old_y;
        std::size_t new_flat = old_flat;

        const double left = phen.value(old_flat - stride, ind_pher);
        const double right = phen.value(old_flat + stride, ind_pher);
        const double up = phen.value(old_flat - 1, ind_pher);
        const double down = phen.value(old_flat + 1, ind_pher);
        const double max_phen = std::max({left, right, up, down});

        if ((choice > m_eps) || (max_phen <= 0.0)) {
            do {
                new_x = old_x;
                new_y = old_y;
                const int d = rand_int32(1, 4, local_seed);
                if (d == 1) new_x -= 1;
                if (d == 2) new_y -= 1;
                if (d == 3) new_x += 1;
                if (d == 4) new_y += 1;
                new_flat = phen.flat_index(new_x, new_y);
            } while (phen.blocked(new_flat, ind_pher));
        } else {
            if (left == max_phen) {
                new_x -= 1;
                new_flat = old_flat - stride;
            } else if (right == max_phen) {
                new_x += 1;
                new_flat = old_flat + stride;
            } else if (up == max_phen) {
                new_y -= 1;
                new_flat = old_flat - 1;
            } else {
                new_y += 1;
                new_flat = old_flat + 1;
            }
        }

        consumed_time += land(new_x, new_y);
        touched_cells.push_back(new_flat);

        x = new_x;
        y = new_y;

        if (new_x == pos_nest.x && new_y == pos_nest.y) {
            if (state == loaded) {
                local_food_counter += 1;
            }
            state = unloaded;
        }

        if (new_x == pos_food.x && new_y == pos_food.y) {
            state = loaded;
        }
    }

    m_pos_x[idx] = x;
    m_pos_y[idx] = y;
    m_state[idx] = state;
    m_seed[idx] = local_seed;
}
