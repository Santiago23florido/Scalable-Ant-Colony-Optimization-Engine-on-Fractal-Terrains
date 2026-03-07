#include "../include/population.hpp"

#include <algorithm>

#include "../include/rand_generator.hpp"

double Population::m_eps = 0.0;

void Population::reserve(std::size_t count) {
    m_pos_x.reserve(count);
    m_pos_y.reserve(count);
    m_state.reserve(count);
    m_seed.reserve(count);
}

void Population::clear() {
    m_pos_x.clear();
    m_pos_y.clear();
    m_state.clear();
    m_seed.clear();
}

void Population::add_ant(const position_t& pos, std::size_t seed) {
    add_ant(pos, seed, unloaded);
}

void Population::add_ant(const position_t& pos, std::size_t seed, std::uint8_t state) {
    m_pos_x.push_back(pos.x);
    m_pos_y.push_back(pos.y);
    m_state.push_back((state == loaded) ? loaded : unloaded);
    m_seed.push_back(seed);
}

void Population::advance_all(pheronome& phen, const fractal_land& land, const position_t& pos_food, const position_t& pos_nest, std::size_t& food_counter) {
    const std::size_t ant_count = size();
    for (std::size_t i = 0; i < ant_count; ++i) {
        advance_one(i, phen, land, pos_food, pos_nest, food_counter);
    }
}

void Population::advance_one(std::size_t idx, pheronome& phen, const fractal_land& land, const position_t& pos_food, const position_t& pos_nest, std::size_t& food_counter) {
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

        const double left = phen.value(old_flat - 1, ind_pher);
        const double right = phen.value(old_flat + 1, ind_pher);
        const double up = phen.value(old_flat - stride, ind_pher);
        const double down = phen.value(old_flat + stride, ind_pher);
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
                new_flat = old_flat - 1;
            } else if (right == max_phen) {
                new_x += 1;
                new_flat = old_flat + 1;
            } else if (up == max_phen) {
                new_y -= 1;
                new_flat = old_flat - stride;
            } else {
                new_y += 1;
                new_flat = old_flat + stride;
            }
        }

        consumed_time += land(new_x, new_y);
        phen.mark_pheronome_flat(new_flat);

        x = new_x;
        y = new_y;

        if (new_x == pos_nest.x && new_y == pos_nest.y) {
            if (state == loaded) {
                food_counter += 1;
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
