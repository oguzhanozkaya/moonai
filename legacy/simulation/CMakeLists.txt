set(MOONAI_SIMULATION_SOURCES
    simulation.cpp
    buffers.cu
    batch.cu
)

add_library(moonai_simulation STATIC ${MOONAI_SIMULATION_SOURCES})

target_include_directories(moonai_simulation PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_simulation PUBLIC
    moonai_build_config
    moonai_core
)

set_property(TARGET moonai_simulation PROPERTY CUDA_ARCHITECTURES native)
