#ifndef TESTS_COMMON_CONFIG_HPP_
#define TESTS_COMMON_CONFIG_HPP_

#include <cstdint>

// clang-format off

constexpr int64_t REPEAT_TIMES = 200;
constexpr int64_t MAX_EPOCHS = 240;
constexpr int64_t INSTANCE_SIZE = 50;
constexpr double NOISE_SIZE = 3;
constexpr double BIAS_SIZE = 0;

constexpr const char* DATA_PATH = "/home/hakula/Tencent/resource_usage_prediction/data";
constexpr const char* OUTPUT_PATH = "/home/hakula/Tencent/resource_usage_prediction/data/input.txt";

constexpr const char* END_MARK = "__END__";

// clang-format on

#endif  // TESTS_COMMON_CONFIG_HPP_
