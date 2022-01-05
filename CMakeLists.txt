cmake_minimum_required(VERSION 3.21)
project(Architecture)

# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_STANDARD 17)

add_executable(Architecture main.cpp checker)
