#pragma once

#include <cstddef>
#include <vector>
#include <mpi.h>

namespace mpi_subdomain {

struct DomainDecomposition {
    
    int rank{0};
    int size{1};

    int global_nx{0};
    int global_ny{0};

    int local_nx{0};
    int local_ny{0};

    int dims_x{1};
    int dims_y{1};
    int coord_x{0};
    int coord_y{0};

    int offset_x{0};
    int offset_y{0};

    // Halo thickness in cells around the local interior.
    int ghost{1};

    // Direct Cartesian neighbors.
    int left_rank{MPI_PROC_NULL};
    int right_rank{MPI_PROC_NULL};
    int up_rank{MPI_PROC_NULL};
    int down_rank{MPI_PROC_NULL};

    // Lookup tables between Cartesian coordinates and ranks.
    std::vector<int> rank_for_coords;
    std::vector<int> rank_coord_x;
    std::vector<int> rank_coord_y;

    // Total local width including halos.
    int stride_x() const { return local_nx + 2 * ghost; }

    // Total local height including halos.
    int stride_y() const { return local_ny + 2 * ghost; }

    // Number of cells in the local halo-padded buffer.
    std::size_t halo_size() const {
        return static_cast<std::size_t>(stride_x()) * static_cast<std::size_t>(stride_y());
    }

    // Flattened index in row-major order for local coordinates.
    std::size_t idx(int local_x, int local_y) const {
        return static_cast<std::size_t>(local_y) * static_cast<std::size_t>(stride_x()) + static_cast<std::size_t>(local_x);
    }
    // Send counts (in cells) per rank for packed gather/scatter.
    std::vector<int> gather_counts_cells() const;

    // Displacements (in cells) per rank for packed gather/scatter.
    std::vector<int> gather_displs_cells() const;

    // True if global x belongs to this local block.
    bool owns_global_x(int gx) const {
        return (gx >= offset_x) && (gx < offset_x + local_nx);
    }

    // True if global y belongs to this local block.
    bool owns_global_y(int gy) const {
        return (gy >= offset_y) && (gy < offset_y + local_ny);
    }

    // True if global (x,y) belongs to this local block.
    bool owns_global(int gx, int gy) const {
        return owns_global_x(gx) && owns_global_y(gy);
    }
    // Convert global x to local x (including halo offset).
    int local_x_from_global(int gx) const {
        return (gx - offset_x) + ghost;
    }

    // Convert global y to local y (including halo offset).
    int local_y_from_global(int gy) const {
        return (gy - offset_y) + ghost;
    }
    // Owner Cartesian coordinate along x for a global x.
    int owner_of_x(int gx) const;

    // Owner Cartesian coordinate along y for a global y.
    int owner_of_y(int gy) const;

    // Owner rank for a global (x,y) cell.
    int owner_of(int gx, int gy) const;

    int rank_at_coords(int cy, int cx) const;
};

// Build a 2D Cartesian decomposition and fill all metadata for each rank.
DomainDecomposition map_decomposed(int global_nx, int global_ny, MPI_Comm comm_world, MPI_Comm& cart_comm);

}