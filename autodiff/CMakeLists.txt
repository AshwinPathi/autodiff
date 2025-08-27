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

# Enable AddressSanitizer if supported
option(ENABLE_ASAN "Enable AddressSanitizer" ON)

if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled")

    # Append flags for Clang/GCC
    set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_FLAGS}")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} ${ASAN_FLAGS}")
endif()
