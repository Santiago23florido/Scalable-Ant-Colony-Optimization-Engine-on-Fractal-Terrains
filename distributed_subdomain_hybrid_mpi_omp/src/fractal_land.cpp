#include "../include/fractal_land.hpp"

#include <algorithm>
#include "../include/rand_generator.hpp"

namespace {
int local_extent_for_coord(int global_n, int dims_n, int coord) {
    const int base = global_n / dims_n;
    const int rem = global_n % dims_n;
    return base + ((coord < rem) ? 1 : 0);
}

int offset_for_coord(int global_n, int dims_n, int coord) {
    const int base = global_n / dims_n;
    const int rem = global_n % dims_n;
    return coord * base + std::min(coord, rem);
}

[[maybe_unused]] int owner_coord_for_value(int value, int global_n, int dims_n) {
    if (value < 0 || value >= global_n) {
        return MPI_PROC_NULL;
    }

    const int base = global_n / dims_n;
    const int rem = global_n % dims_n;

    if (base == 0) {
        return value;
    }

    const int split = (base + 1) * rem;
    if (value < split) {
        return value / (base + 1);
    }
    return rem + (value - split) / base;
}
} 

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

void fractal_land::normalize_land() {
    normalize_land(*this);
}

void fractal_land::normalize_land(fractal_land& land) {
    double max_val = 0.0;
    double min_val = 0.0;

    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i) {
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            max_val = std::max(max_val, land(i, j));
            min_val = std::min(min_val, land(i, j));
        }
    }

    const double delta = max_val - min_val;
    if (delta <= 0.0) {
        return;
    }

    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i) {
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            land(i, j) = (land(i, j) - min_val) / delta;
        }
    }
}

std::vector<double> fractal_land::pack_land_by_rank(const mpi_subdomain::DomainDecomposition& decomp,
                                                    const std::vector<int>& displs) {
    std::vector<double> packed(static_cast<std::size_t>(decomp.global_nx * decomp.global_ny), 1.0);

    for (int r = 0; r < decomp.size; ++r) {
        const std::size_t ridx = static_cast<std::size_t>(r);
        const int cx = decomp.rank_coord_x[ridx];
        const int cy = decomp.rank_coord_y[ridx];

        const int nx_r =
            (cx < 0 || cx >= decomp.dims_x) ? 0 : local_extent_for_coord(decomp.global_nx, decomp.dims_x, cx);
        const int ox_r = (cx < 0 || cx >= decomp.dims_x) ? 0 : offset_for_coord(decomp.global_nx, decomp.dims_x, cx);

        const int ny_r =
            (cy < 0 || cy >= decomp.dims_y) ? 0 : local_extent_for_coord(decomp.global_ny, decomp.dims_y, cy);
        const int oy_r = (cy < 0 || cy >= decomp.dims_y) ? 0 : offset_for_coord(decomp.global_ny, decomp.dims_y, cy);

        const int disp_r = displs[static_cast<std::size_t>(r)];

        for (int ly = 0; ly < ny_r; ++ly) {
            for (int lx = 0; lx < nx_r; ++lx) {
                const std::size_t dst =
                    static_cast<std::size_t>(disp_r) + static_cast<std::size_t>(ly * nx_r + lx);
                packed[dst] = (*this)(static_cast<unsigned long>(ox_r + lx), static_cast<unsigned long>(oy_r + ly));
            }
        }
    }

    return packed;
}
