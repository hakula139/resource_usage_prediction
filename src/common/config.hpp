#ifndef SRC_COMMON_CONFIG_HPP_
#define SRC_COMMON_CONFIG_HPP_

#include <cstdint>

// clang-format off

constexpr int64_t INPUT_SIZE = 1;
constexpr int64_t HIDDEN_SIZE = 200;
constexpr int64_t OUTPUT_SIZE = 1;
constexpr int64_t BATCH_SIZE = 6;
constexpr int64_t MAX_SIZE = 500;
constexpr int64_t N_LAYERS = 3;
constexpr double DROPOUT = 0.2;
constexpr double LEARNING_RATE = 5e-3;
constexpr const char* INPUT_PATH = "data/input.txt";
constexpr const char* OUTPUT_PATH = "data/output.txt";
constexpr const char* PREDICTIONS_FIGURE_PATH = "figures/predictions.pdf";
constexpr const char* LOSS_FIGURE_PATH = "figures/loss.pdf";

// clang-format on

#endif  // SRC_COMMON_CONFIG_HPP_
