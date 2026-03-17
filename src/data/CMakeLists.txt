add_library(moonai_data STATIC
    logger.cpp
    metrics.cpp
)

target_include_directories(moonai_data PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_data PUBLIC
    moonai_core
)
