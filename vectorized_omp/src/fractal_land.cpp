#include "fractal_land.hpp"

#include "rand_generator.hpp"

void fractal_land::compute_subgrid(int log_subgrid_dim, int i_block, int j_block, double deviation,
                                   std::size_t seed) {
    RandomGenerator gen(seed, -deviation, deviation);

    fractal_land& land = *this;
    unsigned long dim_subgrid = 1UL << (log_subgrid_dim);
    unsigned long i_begin = i_block * dim_subgrid;
    unsigned long j_begin = j_block * dim_subgrid;
    int mid_idx = static_cast<int>(dim_subgrid / 2);
    int i_mid = static_cast<int>(i_begin) + mid_idx;
    int j_mid = static_cast<int>(j_begin) + mid_idx;
    int i_end = static_cast<int>(i_begin + dim_subgrid);
    int j_end = static_cast<int>(j_begin + dim_subgrid);

    land(i_mid, j_begin) = 0.5 * (land(i_begin, j_begin) + land(i_end, j_begin)) + mid_idx * gen(i_mid, j_begin);
    land(i_begin, j_mid) = 0.5 * (land(i_begin, j_begin) + land(i_begin, j_end)) + mid_idx * gen(i_begin, j_mid);
    land(i_mid, j_end) = 0.5 * (land(i_begin, j_end) + land(i_end, j_end)) + mid_idx * gen(i_mid, j_end);
    land(i_end, j_mid) = 0.5 * (land(i_end, j_begin) + land(i_end, j_end)) + mid_idx * gen(i_end, j_mid);
    land(i_mid, j_mid) =
        0.25 * (land(i_mid, j_begin) + land(i_begin, j_mid) + land(i_mid, j_end) + land(i_end, j_mid)) +
        mid_idx * gen(i_mid, j_mid);
}

fractal_land::fractal_land(const dim_t& log_size, unsigned long nb_seeds, double deviation, int seed)
    : m_dimensions(0), m_altitude() {
    unsigned long dim_subgrid = 1UL << (log_size);
    m_dimensions = nb_seeds * dim_subgrid + 1;
    container(m_dimensions * m_dimensions).swap(m_altitude);

    RandomGenerator gen(seed, 0., dim_subgrid * deviation);
    fractal_land& land = *this;

    for (dim_t i = 0; i < m_dimensions; i += dim_subgrid) {
        for (dim_t j = 0; j < m_dimensions; j += dim_subgrid) {
            land(i, j) = gen(i, j);
        }
    }

    dim_t level = log_size;
    while (level > 1) {
        level -= 1;
        dim_subgrid /= 2;
        nb_seeds *= 2;
        for (unsigned long i_block = 0; i_block < nb_seeds; ++i_block) {
            for (unsigned long j_block = 0; j_block < nb_seeds; ++j_block) {
                compute_subgrid(static_cast<int>(level), static_cast<int>(i_block), static_cast<int>(j_block),
                                deviation, seed);
            }
        }
    }
}
