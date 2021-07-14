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

    if (dataset.size() == SEQ_LEN) {
      auto data = torch::from_blob(dataset.data(), {SEQ_LEN});
      auto loss = predictor.Train(
          data.slice(0, 0, SEQ_LEN - 1), data[SEQ_LEN - 1]);
      std::cout << "Current loss: " << loss << "\n";
      auto prediction = predictor.Predict(data.slice(0, 1, SEQ_LEN));
      std::cout << "Prediction: " << prediction << "\n";
      dataset.erase(dataset.begin());
    }
  }
}
