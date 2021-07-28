#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

#include "common/config.hpp"
#include "common/utils.hpp"

namespace fs = std::filesystem;

int main() {
  auto seed = 2021;
  std::mt19937 rng(seed);
  std::normal_distribution<double> normal_dist(0.0);

  fs::create_directories(DATA_DIR);
  std::ofstream output_file(OUTPUT_PATH);

  std::vector<double> dataset;
  for (auto x = 0; x < MAX_EPOCHS; ++x) {
    std::vector<double> data{
        // Yesterday
        NormalPdf(x, 140 - MAX_EPOCHS, 100) * 2000,
        NormalPdf(x, 80 - MAX_EPOCHS, 15) * 2000,
        NormalPdf(x, 130 - MAX_EPOCHS, 15) * 3000,
        NormalPdf(x, 190 - MAX_EPOCHS, 10) * 4000,
        NormalPdf(x, 220 - MAX_EPOCHS, 10) * 3000,

        // Today
        NormalPdf(x, 140, 100) * 2000,
        NormalPdf(x, 80, 15) * 2000,
        NormalPdf(x, 130, 15) * 3000,
        NormalPdf(x, 190, 10) * 4000,
        NormalPdf(x, 220, 10) * 3000,

        // Random noise
        normal_dist(rng) * 5,
    };

    auto sum = std::accumulate(data.begin(), data.end(), 0.0);
    dataset.push_back(std::max(sum, 0.0));
  }

  for (auto i = 0; i < 50; ++i) {
    for (auto y : dataset) output_file << round(y) << " ";
    output_file << "\n";
  }
}
