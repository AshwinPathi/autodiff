cmake_minimum_required(VERSION 3.15)
project(my_project VERSION 1.0 LANGUAGES CXX)

# Force clang
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add include directory for headers
include_directories(include)

# Collect all source files (or list them manually)
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create executable
add_executable(my_project ${SOURCES})
