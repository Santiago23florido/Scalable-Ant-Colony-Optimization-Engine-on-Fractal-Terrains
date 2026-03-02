#pragma once

#include <vector>
#include <mpi.h>

#include "../include/subdomain.hpp"

namespace mpi_subdomain {

void set_horizontal_boundary_ghosts(const DomainDecomposition& decomp, std::vector<double>& v1, std::vector<double>& v2, double boundary_value = -1.0);

void exchange_pheromone_halos(const DomainDecomposition& decomp, std::vector<double>& v1, std::vector<double>& v2, MPI_Comm comm);

void exchange_static_halos(const DomainDecomposition& decomp, std::vector<double>& terrain, std::vector<int>& cell_type, MPI_Comm comm);

}  
