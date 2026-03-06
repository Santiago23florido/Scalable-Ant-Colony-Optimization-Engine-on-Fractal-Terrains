#ifndef FRACTAL_LAND_HPP
#define FRACTAL_LAND_HPP

#include <utility>
#include <vector>

class fractal_land {
public:
    using container = std::vector<double>;
    using dim_t = unsigned long;

    fractal_land(const dim_t& log_size, unsigned long nb_seeds, double deviation, int seed = 0);
    fractal_land(const fractal_land&) = delete;
    fractal_land(fractal_land&&) = default;
    ~fractal_land() = default;

    double operator()(unsigned long i, unsigned long j) const { return m_altitude[i + j * m_dimensions]; }
    double& operator()(unsigned long i, unsigned long j) { return m_altitude[i + j * m_dimensions]; }

    dim_t dimensions() const { return m_dimensions; }

private:
    void compute_subgrid(int log_subgrid_dim, int i_block, int j_block, double deviation, std::size_t seed);

    dim_t m_dimensions;
    container m_altitude;
};

#endif
