#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <vector>

#include "population.hpp"
#include "fractal_land.hpp"
#include "pheronome.hpp"
#include "window.hpp"

class Renderer {
public:
    Renderer(const fractal_land& land, const pheronome& phen, const position_t& pos_nest, const position_t& pos_food,
             const Population& ants);
    Renderer(const Renderer&) = delete;
    ~Renderer();

    void display(Window& win, const std::size_t& compteur);

private:
    const fractal_land& m_ref_land;
    SDL_Texture* m_land{nullptr};
    const pheronome& m_ref_phen;
    const position_t& m_pos_nest;
    const position_t& m_pos_food;
    const Population& m_ref_ants;
    std::vector<std::size_t> m_curve;
};

#endif
