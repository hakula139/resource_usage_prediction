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
  std::normal_distribution<double> normal_dist(0.0);

  fs::create_directories(DATA_PATH);
  std::ofstream output_file(OUTPUT_PATH);

  for (auto i = 0; i < 200; ++i) {
    std::mt19937 rng(i);
    // auto bias = normal_dist(rng) * 10;

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
          normal_dist(rng) * 3,
      };

      auto sum = std::accumulate(data.begin(), data.end(), 0);
      auto y = std::max(sum, 0.0);

      for (auto j = 1; j <= 5; ++j) {
        output_file << j << " " << round(y * j) << "\n";
      }
    }
  }

  output_file << END_MARK << std::endl;
}
