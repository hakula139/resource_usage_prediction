#include <torch/torch.h>

#include <iostream>
#include <vector>

#include "common/config.hpp"
#include "predictor/predictor.hpp"

int main() {
  std::vector<int64_t> dataset;
  Predictor predictor;

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
      std::cout << "Loss: " << loss << "\n";

      input = data.slice(0, OUTPUT_SIZE, BATCH_SIZE + OUTPUT_SIZE);
      auto prediction = predictor.Predict(input)[0].item<int64_t>();
      std::cout << "Prediction: " << prediction << "\n";

      dataset.erase(dataset.begin());
    }
  }
}
