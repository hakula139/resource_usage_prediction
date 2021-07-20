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

  std::vector<int64_t> dataset;
  for (auto x = 0; x < MAX_EPOCHS; ++x) {
    auto y1 = NormalPdf(x, 30, 15) * 1000;
    auto y2 = NormalPdf(x, 100, 20) * 2000;
    auto y3 = NormalPdf(x, 160, 15) * 2000;
    dataset.push_back(round(y1 + y2 + y3));
  }

  for (auto i = 0; i < 2; ++i) {
    for (auto j = 1; j <= 10; ++j) {
      for (auto k = 0; k < 2; ++k) {
        for (auto y : dataset) output_file << y / j << " ";
        output_file << "\n";
      }
    }
  }
}
