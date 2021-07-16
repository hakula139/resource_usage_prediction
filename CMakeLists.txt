# ./CMakeLists.txt

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(resource_usage_prediction VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/lib/libtorch")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

find_package(Torch REQUIRED)
find_package(sciplot REQUIRED)

# Parameters
set(CONFIG_INPUT_SIZE 1)
set(CONFIG_HIDDEN_SIZE 200)
set(CONFIG_OUTPUT_SIZE 1)
set(CONFIG_BATCH_SIZE 6)
set(CONFIG_MAX_SIZE 500)
set(CONFIG_N_LAYERS 3)
set(CONFIG_DROPOUT 0.2)
set(CONFIG_LEARNING_RATE 5e-3)
set(CONFIG_INPUT_PATH "data/input.txt")
set(CONFIG_OUTPUT_PATH "data/output.txt")
set(CONFIG_PREDICTIONS_FIGURE_PATH "figures/predictions.pdf")
set(CONFIG_LOSS_FIGURE_PATH "figures/loss.pdf")

configure_file("config.hpp.in" "${CMAKE_SOURCE_DIR}/src/common/config.hpp")

add_subdirectory(src)
add_subdirectory(tests)
