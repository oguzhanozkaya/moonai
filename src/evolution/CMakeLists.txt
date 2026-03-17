add_library(moonai_evolution STATIC
    evolution_manager.cpp
    genome.cpp
    neural_network.cpp
    species.cpp
    mutation.cpp
    crossover.cpp
)

target_include_directories(moonai_evolution PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_evolution PUBLIC
    moonai_core
    moonai_simulation
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(moonai_evolution PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(moonai_evolution PUBLIC MOONAI_OPENMP_ENABLED)
endif()
