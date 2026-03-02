#include "../include/halo.hpp"

#include <array>
#include <algorithm>

namespace mpi_subdomain {

namespace {

constexpr int TAG_DIR_UP = 0;
constexpr int TAG_DIR_DOWN = 1;
constexpr int TAG_DIR_LEFT = 2;
constexpr int TAG_DIR_RIGHT = 3;

// Fill one local row in a double field (decomp: local geometry/indexing, field: mutable buffer, local_y: row id, value: constant to write).
void fill_row_double(const DomainDecomposition& decomp, std::vector<double>& field, int local_y, double value) {
    for (int lx = 0; lx < decomp.stride_x(); ++lx) {
        field[decomp.idx(lx, local_y)] = value;
    }
}

// Fill one local row in an int field (decomp: local geometry/indexing, field: mutable buffer, local_y: row id, value: constant to write).
void fill_row_int(const DomainDecomposition& decomp, std::vector<int>& field, int local_y, int value) {
    for (int lx = 0; lx < decomp.stride_x(); ++lx) {
        field[decomp.idx(lx, local_y)] = value;
    }
}

// Fill one local column in a double field (decomp: local geometry/indexing, field: mutable buffer, local_x: column id, value: constant to write).
void fill_col_double(const DomainDecomposition& decomp, std::vector<double>& field, int local_x, double value) {
    for (int ly = 0; ly < decomp.stride_y(); ++ly) {
        field[decomp.idx(local_x, ly)] = value;
    }
}

// Fill one local column in an int field (decomp: local geometry/indexing, field: mutable buffer, local_x: column id, value: constant to write).
void fill_col_int(const DomainDecomposition& decomp, std::vector<int>& field, int local_x, int value) {
    for (int ly = 0; ly < decomp.stride_y(); ++ly) {
        field[decomp.idx(local_x, ly)] = value;
    }
}

// Exchange one-cell halos for a double field with neighbors (decomp: neighborhood/layout, field: halo-padded data, boundary_value: fallback at physical borders, comm: Cartesian communicator, tag_base: MPI tag offset).
void exchange_halo_double(const DomainDecomposition& decomp, std::vector<double>& field, double boundary_value, MPI_Comm comm, int tag_base) {
    std::vector<double> send_up(static_cast<std::size_t>(decomp.local_nx), boundary_value);
    std::vector<double> send_down(static_cast<std::size_t>(decomp.local_nx), boundary_value);
    std::vector<double> recv_up(static_cast<std::size_t>(decomp.local_nx), boundary_value);
    std::vector<double> recv_down(static_cast<std::size_t>(decomp.local_nx), boundary_value);
    std::vector<double> send_left(static_cast<std::size_t>(decomp.local_ny), boundary_value);
    std::vector<double> send_right(static_cast<std::size_t>(decomp.local_ny), boundary_value);
    std::vector<double> recv_left(static_cast<std::size_t>(decomp.local_ny), boundary_value);
    std::vector<double> recv_right(static_cast<std::size_t>(decomp.local_ny), boundary_value);

    if (decomp.local_nx > 0 && decomp.local_ny > 0) {
        for (int lx = 1; lx <= decomp.local_nx; ++lx) {
            send_up[static_cast<std::size_t>(lx - 1)] = field[decomp.idx(lx, 1)];
            send_down[static_cast<std::size_t>(lx - 1)] = field[decomp.idx(lx, decomp.local_ny)];
        }
        for (int ly = 1; ly <= decomp.local_ny; ++ly) {
            send_left[static_cast<std::size_t>(ly - 1)] = field[decomp.idx(1, ly)];
            send_right[static_cast<std::size_t>(ly - 1)] = field[decomp.idx(decomp.local_nx, ly)];
        }
    }

    std::array<MPI_Request, 8> reqs{};
    int req_count = 0;

    if (decomp.up_rank != MPI_PROC_NULL && decomp.local_nx > 0) {
        MPI_Irecv(recv_up.data(), decomp.local_nx, MPI_DOUBLE, decomp.up_rank, tag_base + TAG_DIR_DOWN, comm, &reqs[req_count++]);
        MPI_Isend(send_up.data(), decomp.local_nx, MPI_DOUBLE, decomp.up_rank, tag_base + TAG_DIR_UP, comm, &reqs[req_count++]);
    }
    if (decomp.down_rank != MPI_PROC_NULL && decomp.local_nx > 0) {
        MPI_Irecv(recv_down.data(), decomp.local_nx, MPI_DOUBLE, decomp.down_rank, tag_base + TAG_DIR_UP, comm, &reqs[req_count++]);
        MPI_Isend(send_down.data(), decomp.local_nx, MPI_DOUBLE, decomp.down_rank, tag_base + TAG_DIR_DOWN, comm, &reqs[req_count++]);
    }
    if (decomp.left_rank != MPI_PROC_NULL && decomp.local_ny > 0) {
        MPI_Irecv(recv_left.data(), decomp.local_ny, MPI_DOUBLE, decomp.left_rank, tag_base + TAG_DIR_RIGHT, comm, &reqs[req_count++]);
        MPI_Isend(send_left.data(), decomp.local_ny, MPI_DOUBLE, decomp.left_rank, tag_base + TAG_DIR_LEFT, comm, &reqs[req_count++]);
    }
    if (decomp.right_rank != MPI_PROC_NULL && decomp.local_ny > 0) {
        MPI_Irecv(recv_right.data(), decomp.local_ny, MPI_DOUBLE, decomp.right_rank, tag_base + TAG_DIR_LEFT, comm, &reqs[req_count++]);
        MPI_Isend(send_right.data(), decomp.local_ny, MPI_DOUBLE, decomp.right_rank, tag_base + TAG_DIR_RIGHT, comm, &reqs[req_count++]);
    }

    if (req_count > 0) {
        MPI_Waitall(req_count, reqs.data(), MPI_STATUSES_IGNORE);
    }

    if (decomp.up_rank == MPI_PROC_NULL || decomp.local_nx == 0 || decomp.local_ny == 0) {
        fill_row_double(decomp, field, 0, boundary_value);
    } else {
        for (int lx = 1; lx <= decomp.local_nx; ++lx) {
            field[decomp.idx(lx, 0)] = recv_up[static_cast<std::size_t>(lx - 1)];
        }
    }

    if (decomp.down_rank == MPI_PROC_NULL || decomp.local_nx == 0 || decomp.local_ny == 0) {
        fill_row_double(decomp, field, decomp.local_ny + 1, boundary_value);
    } else {
        for (int lx = 1; lx <= decomp.local_nx; ++lx) {
            field[decomp.idx(lx, decomp.local_ny + 1)] = recv_down[static_cast<std::size_t>(lx - 1)];
        }
    }

    if (decomp.left_rank == MPI_PROC_NULL || decomp.local_nx == 0 || decomp.local_ny == 0) {
        fill_col_double(decomp, field, 0, boundary_value);
    } else {
        for (int ly = 1; ly <= decomp.local_ny; ++ly) {
            field[decomp.idx(0, ly)] = recv_left[static_cast<std::size_t>(ly - 1)];
        }
    }

    if (decomp.right_rank == MPI_PROC_NULL || decomp.local_nx == 0 || decomp.local_ny == 0) {
        fill_col_double(decomp, field, decomp.local_nx + 1, boundary_value);
    } else {
        for (int ly = 1; ly <= decomp.local_ny; ++ly) {
            field[decomp.idx(decomp.local_nx + 1, ly)] = recv_right[static_cast<std::size_t>(ly - 1)];
        }
    }
}

// Exchange one-cell halos for an int field with neighbors (decomp: neighborhood/layout, field: halo-padded data, boundary_value: fallback at physical borders, comm: Cartesian communicator, tag_base: MPI tag offset).
void exchange_halo_int(const DomainDecomposition& decomp, std::vector<int>& field, int boundary_value, MPI_Comm comm, int tag_base) {
    std::vector<int> send_up(static_cast<std::size_t>(decomp.local_nx), boundary_value);
    std::vector<int> send_down(static_cast<std::size_t>(decomp.local_nx), boundary_value);
    std::vector<int> recv_up(static_cast<std::size_t>(decomp.local_nx), boundary_value);
    std::vector<int> recv_down(static_cast<std::size_t>(decomp.local_nx), boundary_value);
    std::vector<int> send_left(static_cast<std::size_t>(decomp.local_ny), boundary_value);
    std::vector<int> send_right(static_cast<std::size_t>(decomp.local_ny), boundary_value);
    std::vector<int> recv_left(static_cast<std::size_t>(decomp.local_ny), boundary_value);
    std::vector<int> recv_right(static_cast<std::size_t>(decomp.local_ny), boundary_value);

    if (decomp.local_nx > 0 && decomp.local_ny > 0) {
        for (int lx = 1; lx <= decomp.local_nx; ++lx) {
            send_up[static_cast<std::size_t>(lx - 1)] = field[decomp.idx(lx, 1)];
            send_down[static_cast<std::size_t>(lx - 1)] = field[decomp.idx(lx, decomp.local_ny)];
        }
        for (int ly = 1; ly <= decomp.local_ny; ++ly) {
            send_left[static_cast<std::size_t>(ly - 1)] = field[decomp.idx(1, ly)];
            send_right[static_cast<std::size_t>(ly - 1)] = field[decomp.idx(decomp.local_nx, ly)];
        }
    }

    std::array<MPI_Request, 8> reqs{};
    int req_count = 0;

    if (decomp.up_rank != MPI_PROC_NULL && decomp.local_nx > 0) {
        MPI_Irecv(recv_up.data(), decomp.local_nx, MPI_INT, decomp.up_rank, tag_base + TAG_DIR_DOWN, comm, &reqs[req_count++]);
        MPI_Isend(send_up.data(), decomp.local_nx, MPI_INT, decomp.up_rank, tag_base + TAG_DIR_UP, comm, &reqs[req_count++]);
    }
    if (decomp.down_rank != MPI_PROC_NULL && decomp.local_nx > 0) {
        MPI_Irecv(recv_down.data(), decomp.local_nx, MPI_INT, decomp.down_rank, tag_base + TAG_DIR_UP, comm, &reqs[req_count++]);
        MPI_Isend(send_down.data(), decomp.local_nx, MPI_INT, decomp.down_rank, tag_base + TAG_DIR_DOWN, comm, &reqs[req_count++]);
    }
    if (decomp.left_rank != MPI_PROC_NULL && decomp.local_ny > 0) {
        MPI_Irecv(recv_left.data(), decomp.local_ny, MPI_INT, decomp.left_rank, tag_base + TAG_DIR_RIGHT, comm, &reqs[req_count++]);
        MPI_Isend(send_left.data(), decomp.local_ny, MPI_INT, decomp.left_rank, tag_base + TAG_DIR_LEFT, comm, &reqs[req_count++]);
    }
    if (decomp.right_rank != MPI_PROC_NULL && decomp.local_ny > 0) {
        MPI_Irecv(recv_right.data(), decomp.local_ny, MPI_INT, decomp.right_rank, tag_base + TAG_DIR_LEFT, comm, &reqs[req_count++]);
        MPI_Isend(send_right.data(), decomp.local_ny, MPI_INT, decomp.right_rank, tag_base + TAG_DIR_RIGHT, comm, &reqs[req_count++]);
    }

    if (req_count > 0) {
        MPI_Waitall(req_count, reqs.data(), MPI_STATUSES_IGNORE);
    }

    if (decomp.up_rank == MPI_PROC_NULL || decomp.local_nx == 0 || decomp.local_ny == 0) {
        fill_row_int(decomp, field, 0, boundary_value);
    } else {
        for (int lx = 1; lx <= decomp.local_nx; ++lx) {
            field[decomp.idx(lx, 0)] = recv_up[static_cast<std::size_t>(lx - 1)];
        }
    }

    if (decomp.down_rank == MPI_PROC_NULL || decomp.local_nx == 0 || decomp.local_ny == 0) {
        fill_row_int(decomp, field, decomp.local_ny + 1, boundary_value);
    } else {
        for (int lx = 1; lx <= decomp.local_nx; ++lx) {
            field[decomp.idx(lx, decomp.local_ny + 1)] = recv_down[static_cast<std::size_t>(lx - 1)];
        }
    }

    if (decomp.left_rank == MPI_PROC_NULL || decomp.local_nx == 0 || decomp.local_ny == 0) {
        fill_col_int(decomp, field, 0, boundary_value);
    } else {
        for (int ly = 1; ly <= decomp.local_ny; ++ly) {
            field[decomp.idx(0, ly)] = recv_left[static_cast<std::size_t>(ly - 1)];
        }
    }

    if (decomp.right_rank == MPI_PROC_NULL || decomp.local_nx == 0 || decomp.local_ny == 0) {
        fill_col_int(decomp, field, decomp.local_nx + 1, boundary_value);
    } else {
        for (int ly = 1; ly <= decomp.local_ny; ++ly) {
            field[decomp.idx(decomp.local_nx + 1, ly)] = recv_right[static_cast<std::size_t>(ly - 1)];
        }
    }
}

}  

// Enforce fixed boundary values on pheromone ghost borders when a neighbor is missing (decomp: neighborhood/layout, v1/v2: pheromone channels, boundary_value: border sentinel).
void set_horizontal_boundary_ghosts(const DomainDecomposition& decomp, std::vector<double>& v1, std::vector<double>& v2, double boundary_value) {
    if (decomp.left_rank == MPI_PROC_NULL) {
        for (int ly = 0; ly < decomp.stride_y(); ++ly) {
            v1[decomp.idx(0, ly)] = boundary_value;
            v2[decomp.idx(0, ly)] = boundary_value;
        }
    }
    if (decomp.right_rank == MPI_PROC_NULL) {
        for (int ly = 0; ly < decomp.stride_y(); ++ly) {
            v1[decomp.idx(decomp.local_nx + 1, ly)] = boundary_value;
            v2[decomp.idx(decomp.local_nx + 1, ly)] = boundary_value;
        }
    }
    if (decomp.up_rank == MPI_PROC_NULL) {
        for (int lx = 0; lx < decomp.stride_x(); ++lx) {
            v1[decomp.idx(lx, 0)] = boundary_value;
            v2[decomp.idx(lx, 0)] = boundary_value;
        }
    }
    if (decomp.down_rank == MPI_PROC_NULL) {
        for (int lx = 0; lx < decomp.stride_x(); ++lx) {
            v1[decomp.idx(lx, decomp.local_ny + 1)] = boundary_value;
            v2[decomp.idx(lx, decomp.local_ny + 1)] = boundary_value;
        }
    }
}

// Exchange halos for both pheromone channels and then apply physical-border ghosts (decomp: neighborhood/layout, v1/v2: halo-padded channels, comm: Cartesian communicator).
void exchange_pheromone_halos(const DomainDecomposition& decomp, std::vector<double>& v1, std::vector<double>& v2, MPI_Comm comm) {
    exchange_halo_double(decomp, v1, -1.0, comm, 700);
    exchange_halo_double(decomp, v2, -1.0, comm, 710);
    set_horizontal_boundary_ghosts(decomp, v1, v2, -1.0);
}

// Exchange halos for static maps and clamp physical borders to obstacle/default values (decomp: neighborhood/layout, terrain: movement-cost map, cell_type: occupancy map, comm: Cartesian communicator).
void exchange_static_halos(const DomainDecomposition& decomp, std::vector<double>& terrain, std::vector<int>& cell_type, MPI_Comm comm) {
    exchange_halo_double(decomp, terrain, 1.0, comm, 720);
    exchange_halo_int(decomp, cell_type, -1, comm, 730);

    if (decomp.left_rank == MPI_PROC_NULL) {
        for (int ly = 0; ly < decomp.stride_y(); ++ly) {
            terrain[decomp.idx(0, ly)] = 1.0;
            cell_type[decomp.idx(0, ly)] = -1;
        }
    }
    if (decomp.right_rank == MPI_PROC_NULL) {
        for (int ly = 0; ly < decomp.stride_y(); ++ly) {
            terrain[decomp.idx(decomp.local_nx + 1, ly)] = 1.0;
            cell_type[decomp.idx(decomp.local_nx + 1, ly)] = -1;
        }
    }
    if (decomp.up_rank == MPI_PROC_NULL) {
        for (int lx = 0; lx < decomp.stride_x(); ++lx) {
            terrain[decomp.idx(lx, 0)] = 1.0;
            cell_type[decomp.idx(lx, 0)] = -1;
        }
    }
    if (decomp.down_rank == MPI_PROC_NULL) {
        for (int lx = 0; lx < decomp.stride_x(); ++lx) {
            terrain[decomp.idx(lx, decomp.local_ny + 1)] = 1.0;
            cell_type[decomp.idx(lx, decomp.local_ny + 1)] = -1;
        }
    }
}

}
