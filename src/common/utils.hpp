#ifndef SRC_COMMON_UTILS_HPP_
#define SRC_COMMON_UTILS_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "types.hpp"

void PlotPredictions(
    const std::string& figure_path,
    const std::vector<int64_t>& x1,
    const std::vector<value_t>& y1,
    const std::vector<int64_t>& x2,
    const std::vector<value_t>& y2);

void PlotLoss(
    const std::string& figure_path,
    const std::vector<int64_t>& x1,
    const std::vector<double>& y1,
    const std::vector<int64_t>& x2,
    const std::vector<double>& y2);

#endif  // SRC_COMMON_UTILS_HPP_
