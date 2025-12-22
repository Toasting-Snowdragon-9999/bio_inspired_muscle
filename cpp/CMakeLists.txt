cmake_minimum_required(VERSION 4.0)
project(bio_inspired_locomotion)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

SET(cpg_sources

)

SET(pd_sources

)

SET(sources

)

add_executable(bio_inspired_locomotion project_main.cpp ${cpg_sources} ${pd_sources} ${shared_sources})

add_executable(cpg cpg_main.cpp ${cpg_sources} ${shared_sources})

add_executable(pd_controller pd_main.cpp ${pd_sources} ${shared_sources})

target_include_directories(cpg PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)