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
set(CONFIG_HIDDEN_SIZE 50)
set(CONFIG_OUTPUT_SIZE 1)
set(CONFIG_BATCH_SIZE 50)
set(CONFIG_MAX_SIZE 1000)
set(CONFIG_N_LAYERS 2)
set(CONFIG_DROPOUT 0.3)
set(CONFIG_LEARNING_RATE 4e-3)

set(CONFIG_DATA_DIR "data")
set(CONFIG_INPUT_PATH "${CONFIG_DATA_DIR}/input.txt")
set(CONFIG_OUTPUT_PATH "${CONFIG_DATA_DIR}/output.txt")
set(CONFIG_FIGURE_DIR "figures")
set(CONFIG_PREDICTIONS_FIGURE_PATH "${CONFIG_FIGURE_DIR}/predictions.svg")
set(CONFIG_TRAIN_LOSS_FIGURE_PATH "${CONFIG_FIGURE_DIR}/train_loss.svg")
set(CONFIG_VALID_LOSS_FIGURE_PATH "${CONFIG_FIGURE_DIR}/valid_loss.svg")

configure_file("config.hpp.in" "${CMAKE_SOURCE_DIR}/src/common/config.hpp")

add_subdirectory(src)
add_subdirectory(tests)
