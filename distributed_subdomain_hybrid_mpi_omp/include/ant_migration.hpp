#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <mpi.h>

#include "../include/population.hpp"
#include "../include/basic_types.hpp"
#include "../include/subdomain.hpp"

namespace mpi_subdomain {

struct TransitAnt {
    int x;
    int y;
    std::uint64_t seed;
    double consumed;
    std::uint8_t loaded;
    std::uint8_t pending_deposit;
};

struct StepContext {
    const std::vector<double>& terrain;

    const std::vector<double>& cur_v1;
    const std::vector<double>& cur_v2;
    std::vector<double>& next_v1;
    std::vector<double>& next_v2;

    position_t pos_food;
    position_t pos_nest;

    double alpha;
    double eps;
};

struct StepResult {
    std::size_t food_collected_local{0};
    double move_local_time{0.0};
    double migration_time{0.0};
};

// Distributes initial ants from rank 0 to their owning subdomains.
void distribute_initial_ants(const DomainDecomposition& decomp, int nb_ants, std::size_t global_seed, Population& local_ants, MPI_Comm comm);

// Advances local ants one step and migrates ants crossing subdomain borders.
StepResult advance_ants_with_migration(const DomainDecomposition& decomp, const StepContext& ctx, Population& ants, MPI_Comm comm);

}  // namespace mpi_subdomain
