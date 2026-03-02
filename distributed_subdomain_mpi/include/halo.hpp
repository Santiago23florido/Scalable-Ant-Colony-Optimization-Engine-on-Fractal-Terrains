#pragma once

#include <array>
#include <vector>
#include <mpi.h>

#include "../include/subdomain.hpp"

namespace mpi_subdomain {

struct PheromoneHaloExchange {
    std::vector<double> send_up;
    std::vector<double> send_down;
    std::vector<double> send_left;
    std::vector<double> send_right;
    std::vector<double> recv_up;
    std::vector<double> recv_down;
    std::vector<double> recv_left;
    std::vector<double> recv_right;
    std::array<MPI_Request, 8> requests{};
    int request_count{0};
};

void set_horizontal_boundary_ghosts(const DomainDecomposition& decomp, std::vector<double>& v1, std::vector<double>& v2, double boundary_value = -1.0);

void begin_pheromone_halo_exchange(const DomainDecomposition& decomp, const std::vector<double>& v1, const std::vector<double>& v2, PheromoneHaloExchange& exchange, MPI_Comm comm);

void end_pheromone_halo_exchange(const DomainDecomposition& decomp, std::vector<double>& v1, std::vector<double>& v2, PheromoneHaloExchange& exchange, double boundary_value = -1.0);

void exchange_pheromone_halos(const DomainDecomposition& decomp, std::vector<double>& v1, std::vector<double>& v2, MPI_Comm comm);

void exchange_static_halos(const DomainDecomposition& decomp, std::vector<double>& terrain, std::vector<int>& cell_type, MPI_Comm comm);

}  
