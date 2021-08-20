#ifndef SRC_COMMON_CONFIG_HPP_
#define SRC_COMMON_CONFIG_HPP_

#include <cstdint>

// clang-format off

#define VERBOSE 1

constexpr int64_t HIDDEN_SIZE = 10;
constexpr int64_t OUTPUT_SIZE = 1;
constexpr int64_t MAX_SIZE = 1000;
constexpr int64_t SEQ_LEN = 15;
constexpr int64_t WINDOW_SIZE = 300;
constexpr int64_t N_LAYERS = 1;
constexpr double DROPOUT = 0.3;
constexpr double LEARNING_RATE = 1e-2;
constexpr double MIN_LEARNING_RATE = 1e-3;
constexpr double LOSS_THRESHOLD = 0.1;

constexpr int64_t PLOT_STEP = 500;
constexpr const char* DATA_PATH = "/home/hakula/Tencent/resource_usage_prediction/data";
constexpr const char* INPUT_PATH = "/home/hakula/Tencent/resource_usage_prediction/data/input.txt";
constexpr const char* OUTPUT_PATH = "/home/hakula/Tencent/resource_usage_prediction/data/output.txt";
constexpr const char* FIGURE_PATH = "/home/hakula/Tencent/resource_usage_prediction/figures";
constexpr const char* LOG_PATH = "/home/hakula/Tencent/resource_usage_prediction/logs";

constexpr int64_t WAIT_TIME = 200;
constexpr const char* END_MARK = "__END__";

// clang-format on

#endif  // SRC_COMMON_CONFIG_HPP_
