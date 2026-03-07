#include "../include/subdomain.hpp"

#include <algorithm>
#include <array>

  // namespace mpi_subdomain
namespace mpi_subdomain {

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

int owner_coord_for_value(int value, int global_n, int dims_n) {
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

int DomainDecomposition::rank_at_coords(int cy, int cx) const {
    if (cy < 0 || cy >= dims_y || cx < 0 || cx >= dims_x) {
        return MPI_PROC_NULL;
    }
    return rank_for_coords[static_cast<std::size_t>(cy * dims_x + cx)];
}

int DomainDecomposition::owner_of_x(int gx) const {
    return owner_coord_for_value(gx, global_nx, dims_x);
}

int DomainDecomposition::owner_of_y(int gy) const {
    return owner_coord_for_value(gy, global_ny, dims_y);
}

int DomainDecomposition::owner_of(int gx, int gy) const {
    const int cx = owner_of_x(gx);
    const int cy = owner_of_y(gy);
    if (cx == MPI_PROC_NULL || cy == MPI_PROC_NULL) {
        return MPI_PROC_NULL;
    }
    return rank_at_coords(cy, cx);
}

std::vector<int> DomainDecomposition::gather_counts_cells() const {
    std::vector<int> counts(size, 0);
    for (int r = 0; r < size; ++r) {
        const std::size_t idx = static_cast<std::size_t>(r);
        const int cx = rank_coord_x[idx];
        const int cy = rank_coord_y[idx];
        const int nx = (cx < 0 || cx >= dims_x) ? 0 : local_extent_for_coord(global_nx, dims_x, cx);
        const int ny = (cy < 0 || cy >= dims_y) ? 0 : local_extent_for_coord(global_ny, dims_y, cy);
        counts[idx] = nx * ny;
    }
    return counts;
}

std::vector<int> DomainDecomposition::gather_displs_cells() const {
    std::vector<int> displs(size, 0);
    int running = 0;
    for (int r = 0; r < size; ++r) {
        const std::size_t idx = static_cast<std::size_t>(r);
        displs[idx] = running;

        const int cx = rank_coord_x[idx];
        const int cy = rank_coord_y[idx];
        const int nx = (cx < 0 || cx >= dims_x) ? 0 : local_extent_for_coord(global_nx, dims_x, cx);
        const int ny = (cy < 0 || cy >= dims_y) ? 0 : local_extent_for_coord(global_ny, dims_y, cy);
        running += nx * ny;
    }
    return displs;
}

DomainDecomposition map_decomposed(int global_nx, int global_ny, MPI_Comm comm_world, MPI_Comm& cart_comm) {
    int world_size = 1;
    MPI_Comm_size(comm_world, &world_size);

    std::array<int, 2> dims{0, 0};
    MPI_Dims_create(world_size, 2, dims.data());

    std::array<int, 2> periods{0, 0};
    MPI_Cart_create(comm_world, 2, dims.data(), periods.data(), 0, &cart_comm);

    DomainDecomposition decomp;
    MPI_Comm_rank(cart_comm, &decomp.rank);
    MPI_Comm_size(cart_comm, &decomp.size);

    decomp.global_nx = global_nx;
    decomp.global_ny = global_ny;

    decomp.dims_y = dims[0];
    decomp.dims_x = dims[1];

    std::array<int, 2> coords{0, 0};
    MPI_Cart_coords(cart_comm, decomp.rank, 2, coords.data());
    decomp.coord_y = coords[0];
    decomp.coord_x = coords[1];

    decomp.local_nx = (decomp.coord_x < 0 || decomp.coord_x >= decomp.dims_x)
                          ? 0
                          : local_extent_for_coord(decomp.global_nx, decomp.dims_x, decomp.coord_x);
    decomp.local_ny = (decomp.coord_y < 0 || decomp.coord_y >= decomp.dims_y)
                          ? 0
                          : local_extent_for_coord(decomp.global_ny, decomp.dims_y, decomp.coord_y);
    decomp.offset_x = (decomp.coord_x < 0 || decomp.coord_x >= decomp.dims_x)
                          ? 0
                          : offset_for_coord(decomp.global_nx, decomp.dims_x, decomp.coord_x);
    decomp.offset_y = (decomp.coord_y < 0 || decomp.coord_y >= decomp.dims_y)
                          ? 0
                          : offset_for_coord(decomp.global_ny, decomp.dims_y, decomp.coord_y);

    MPI_Cart_shift(cart_comm, 0, 1, &decomp.up_rank, &decomp.down_rank);
    MPI_Cart_shift(cart_comm, 1, 1, &decomp.left_rank, &decomp.right_rank);

    decomp.rank_for_coords.resize(static_cast<std::size_t>(decomp.dims_x * decomp.dims_y), MPI_PROC_NULL);
    for (int cy = 0; cy < decomp.dims_y; ++cy) {
        for (int cx = 0; cx < decomp.dims_x; ++cx) {
            std::array<int, 2> c{cy, cx};
            int r = MPI_PROC_NULL;
            MPI_Cart_rank(cart_comm, c.data(), &r);
            decomp.rank_for_coords[static_cast<std::size_t>(cy * decomp.dims_x + cx)] = r;
        }
    }

    decomp.rank_coord_x.resize(static_cast<std::size_t>(decomp.size), 0);
    decomp.rank_coord_y.resize(static_cast<std::size_t>(decomp.size), 0);
    for (int r = 0; r < decomp.size; ++r) {
        std::array<int, 2> c{0, 0};
        MPI_Cart_coords(cart_comm, r, 2, c.data());
        decomp.rank_coord_y[static_cast<std::size_t>(r)] = c[0];
        decomp.rank_coord_x[static_cast<std::size_t>(r)] = c[1];
    }

    return decomp;
}

}
