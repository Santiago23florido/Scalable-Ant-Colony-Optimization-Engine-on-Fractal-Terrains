cmake_minimum_required(VERSION 3.16)

# Prefer the MPI wrapper when no compiler is explicitly provided.
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(DMPI_MPICXX NAMES mpic++ mpicxx)
    if(DMPI_MPICXX)
        set(CMAKE_CXX_COMPILER "${DMPI_MPICXX}" CACHE FILEPATH "C++ compiler" FORCE)
    endif()
endif()

project(distributed_subdomain_hybrid_mpi_omp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DMPI_ENABLE_WARNINGS "Enable strict compile warnings" ON)

find_package(MPI QUIET COMPONENTS CXX)
find_package(OpenMP QUIET)
find_package(SDL2 REQUIRED)

add_executable(distributed_subdomain_hybrid_mpi_omp_sim
    src/ant_simu.cpp
    src/ant_migration.cpp
    src/population.cpp
    src/fractal_land.cpp
    src/halo.cpp
    src/renderer.cpp
    src/subdomain.cpp
    src/window.cpp
)

target_include_directories(distributed_subdomain_hybrid_mpi_omp_sim
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(distributed_subdomain_hybrid_mpi_omp_sim
    PRIVATE
)

if(TARGET MPI::MPI_CXX)
    target_link_libraries(distributed_subdomain_hybrid_mpi_omp_sim PRIVATE MPI::MPI_CXX)
else()
    message(STATUS "MPI::MPI_CXX not found: relying on the selected MPI wrapper compiler for MPI headers/libs.")
endif()

if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(distributed_subdomain_hybrid_mpi_omp_sim PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP::OpenMP_CXX not found: building hybrid target without OpenMP support.")
endif()

if(TARGET SDL2::SDL2)
    target_link_libraries(distributed_subdomain_hybrid_mpi_omp_sim PRIVATE SDL2::SDL2)
else()
    target_include_directories(distributed_subdomain_hybrid_mpi_omp_sim PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(distributed_subdomain_hybrid_mpi_omp_sim PRIVATE ${SDL2_LIBRARIES})
endif()

if(DMPI_ENABLE_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(distributed_subdomain_hybrid_mpi_omp_sim PRIVATE -Wall -Wextra -Wpedantic)
    elseif(MSVC)
        target_compile_options(distributed_subdomain_hybrid_mpi_omp_sim PRIVATE /W4)
    endif()
endif()

set_target_properties(distributed_subdomain_hybrid_mpi_omp_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build"
)
