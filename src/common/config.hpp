#ifndef SRC_COMMON_CONFIG_HPP_
#define SRC_COMMON_CONFIG_HPP_

#include <cstdint>

// clang-format off

constexpr int64_t MAX_SIZE = 500;
constexpr int64_t HIDDEN_SIZE = 50;
constexpr int64_t SEQ_LEN = 5;
constexpr int64_t N_LAYERS = 2;
constexpr double DROPOUT = 0.2;
constexpr double LEARNING_RATE = 5e-3;

// clang-format on

#endif  // SRC_COMMON_CONFIG_HPP_
