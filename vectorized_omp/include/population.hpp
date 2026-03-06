#ifndef POPULATION_HPP
#define POPULATION_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "basic_types.hpp"
#include "fractal_land.hpp"
#include "pheronome.hpp"

class Population {
public:
    static constexpr std::uint8_t unloaded = 0;
    static constexpr std::uint8_t loaded = 1;

    Population() = default;

    void reserve(std::size_t count);
    void add_ant(const position_t& pos, std::size_t seed);

    std::size_t size() const { return m_pos_x.size(); }
    int pos_x(std::size_t idx) const { return m_pos_x[idx]; }
    int pos_y(std::size_t idx) const { return m_pos_y[idx]; }

    static void set_exploration_coef(double eps) { m_eps = eps; }

    void advance_all(pheronome& phen, const fractal_land& land, const position_t& pos_food, const position_t& pos_nest,
                     std::size_t& food_counter);

private:
    void advance_one(std::size_t idx, pheronome& phen, const fractal_land& land, const position_t& pos_food,
                     const position_t& pos_nest, std::size_t& local_food_counter,
                     std::vector<pheronome::size_t>& touched_cells);

    static double m_eps;
    std::vector<int> m_pos_x;
    std::vector<int> m_pos_y;
    std::vector<std::uint8_t> m_state;
    std::vector<std::size_t> m_seed;
};

#endif
