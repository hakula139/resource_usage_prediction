#ifndef SRC_COMMON_CONFIG_HPP_
#define SRC_COMMON_CONFIG_HPP_

#include <cstdint>

// clang-format off

constexpr int64_t INPUT_SIZE = 1;
constexpr int64_t HIDDEN_SIZE = 50;
constexpr int64_t OUTPUT_SIZE = 1;
constexpr int64_t BATCH_SIZE = 50;
constexpr int64_t MAX_SIZE = 1000;
constexpr int64_t N_LAYERS = 2;
constexpr double DROPOUT = 0.3;
constexpr double LEARNING_RATE = 5e-3;

constexpr const char* DATA_DIR = "data";
constexpr const char* INPUT_PATH = "data/input.txt";
constexpr const char* OUTPUT_PATH = "data/output.txt";
constexpr const char* FIGURE_DIR = "figures";
constexpr const char* PREDICTIONS_FIGURE_PATH = "figures/predictions.svg";
constexpr const char* TRAIN_LOSS_FIGURE_PATH = "figures/train_loss.svg";
constexpr const char* VALID_LOSS_FIGURE_PATH = "figures/valid_loss.svg";

// clang-format on

#endif  // SRC_COMMON_CONFIG_HPP_
