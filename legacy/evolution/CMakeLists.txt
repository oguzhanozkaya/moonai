set(MOONAI_EVOLUTION_SOURCES
    evolution_manager.cpp
    genome.cpp
    neural_network.cpp
    species.cpp
    mutation.cpp
    crossover.cpp
    network_cache.cpp
    inference_cache.cu
)

add_library(moonai_evolution STATIC ${MOONAI_EVOLUTION_SOURCES})

target_include_directories(moonai_evolution PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_evolution PUBLIC
    moonai_build_config
    moonai_core
    moonai_simulation
)

set_property(TARGET moonai_evolution PROPERTY CUDA_ARCHITECTURES native)
