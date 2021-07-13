# ./CMakeLists.txt

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(resource_usage_prediction VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

find_package(Torch REQUIRED)

add_subdirectory(src)
