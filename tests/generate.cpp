#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include "common/config.hpp"
#include "common/utils.hpp"

namespace fs = std::filesystem;

int main() {
  fs::create_directories(DATA_DIR);

  std::ofstream output_file(OUTPUT_PATH);

  std::vector<double> dataset;
  for (auto x = 0; x < MAX_EPOCHS; ++x) {
    auto y1 = NormalPdf(x, 30, 15) * 1000;
    auto y2 = NormalPdf(x, 90, 20) * 2000;
    auto y3 = NormalPdf(x, 160, 15) * 3000;
    dataset.push_back(y1 + y2 + y3);
  }

  for (auto i = 0; i < 5; ++i) {
    for (auto j = 0; j < 3; ++j) {
      for (auto k = 1; k <= 3; ++k) {
        for (auto y : dataset) output_file << round(y / k) << " ";
        output_file << "\n";
      }
    }
  }
}
