#include <cmath>
#include <fstream>
#include <vector>

#include "common/config.hpp"
#include "common/utils.hpp"

int main() {
  std::ofstream output_file(OUTPUT_PATH);

  std::vector<int64_t> dataset;
  for (auto x = 0; x < MAX_EPOCHS; ++x) {
    auto y1 = NormalPdf(x, 30, 5) * 600;
    auto y2 = NormalPdf(x, 70, 10) * 1000;
    auto y3 = NormalPdf(x, 100, 8) * 800;
    dataset.push_back(round(y1 + y2 + y3));
  }

  for (auto i = 0; i < 5; ++i) {
    for (auto y : dataset) output_file << y << " ";
    output_file << "\n";
  }
}
