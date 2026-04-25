set(MOONAI_METRICS_SOURCES
    logger.cpp
    metrics.cpp
)

add_library(moonai_metrics STATIC ${MOONAI_METRICS_SOURCES})

target_include_directories(moonai_metrics PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(moonai_metrics PUBLIC
    moonai_build_config
    moonai_core
    moonai_evolution
)
