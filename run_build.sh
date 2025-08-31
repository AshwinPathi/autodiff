#!/bin/bash

SKIP_TESTS=0
while getopts "s" opt; do
    case $opt in
        s) SKIP_TESTS=1 ;;
        *) echo "Usage: $0 [-s] (s flag skips tests)" && exit 1 ;;
    esac
done

cwd=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set up cmake build directory if it isn't already setup
build_directory="${cwd}/build/"

if ! [ -d "${build_directory}" ]; then
    echo "Build directory: ${build_directory} does not exist. Making one."
    mkdir -p ${build_directory}
fi

# format files
if [ -x "$(clang-format --version)" ]; then
    find . \( -name "*.cpp" -o -name "*.cc" -o -name "*.c" -o -name "*.h" -o -name "*.hpp" \) -exec clang-format -i {} \;
fi

# build with cmake
cd "${build_directory}"
# cmake -Wno-dev ..
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON -Wno-dev ..
cmake --build .
if [ $SKIP_TESTS -eq 0 ]; then
    ctest --output-on-failure
else
    echo "Skipping tests"
fi

# copy to root for convenience
cp my_project ../run
