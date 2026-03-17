add_library(moonai_simulation STATIC
    simulation_manager.cpp
    environment.cpp
    agent.cpp
    predator.cpp
    prey.cpp
    physics.cpp
    spatial_grid.cpp
)

target_include_directories(moonai_simulation PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_simulation PUBLIC
    moonai_core
)
