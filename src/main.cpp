#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "common/config.hpp"
#include "predictor/predictor.hpp"

int main() {
  std::vector<int64_t> dataset;
  Predictor predictor;
  std::ofstream output(OUTPUT_PATH);

  std::cout << "Server started.\n";

  while (true) {
    int64_t cur_data;
    std::cin >> cur_data;
    if (cur_data < 0) break;
    dataset.push_back(cur_data);

    if (dataset.size() == BATCH_SIZE + OUTPUT_SIZE) {
      auto data = torch::tensor(dataset);

      auto input = data.slice(0, 0, BATCH_SIZE);
      auto expected = data.slice(0, BATCH_SIZE, BATCH_SIZE + OUTPUT_SIZE);
      auto loss = predictor.Train(input, expected);

      input = data.slice(0, OUTPUT_SIZE, BATCH_SIZE + OUTPUT_SIZE);
      auto prediction = predictor.Predict(input)[0].item<int64_t>();
      prediction = prediction > 0 ? prediction : 0;
      std::cout << "> " << prediction << " | Loss: " << loss << "\n";
      output << prediction << " ";

      dataset.erase(dataset.begin());
    }
  }

  output.close();
}
