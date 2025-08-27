cmake_minimum_required(VERSION 3.10)

# Project name
project(MyProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add source files
file(GLOB_RECURSE SOURCES src/*.cpp include/*.hpp)

# Include directories
include_directories(include)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries (if any)
# target_link_libraries(${PROJECT_NAME} library_name)