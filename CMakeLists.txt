cmake_minimum_required(VERSION 3.21)
project(ContrastEnhancer)

set(CMAKE_CXX_STANDARD 17)

add_executable(hw5 main.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(hw5 PRIVATE OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS -O3)

set(CMAKE_CXX_FLAGS -Wall)
set(CMAKE_CXX_FLAGS -Wextra)
set(CMAKE_CXX_FLAGS -pedantic)
