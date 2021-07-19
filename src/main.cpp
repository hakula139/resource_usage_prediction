#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "common/config.hpp"
#include "common/utils.hpp"
#include "predictor/predictor.hpp"

int main() {
  std::vector<int64_t> dataset;
  Predictor predictor;
  std::ifstream input_file(INPUT_PATH);
  std::ofstream output_file(OUTPUT_PATH);

  // Points for plotting
  std::vector<int64_t> expected_x, expected_y;
  std::vector<int64_t> prediction_x, prediction_y;
  std::vector<int64_t> loss_x, loss_y;

  std::cout << "Server started.\n";

  for (int64_t epoch = 1; !input_file.eof(); ++epoch) {
    int64_t cur_data;
    input_file >> cur_data;
    if (cur_data < 0) break;
    dataset.push_back(cur_data);

    expected_x.push_back(epoch);
    expected_y.push_back(cur_data);

    if (dataset.size() == BATCH_SIZE + OUTPUT_SIZE) {
      auto data = torch::tensor(dataset);

      auto input = data.slice(0, 0, BATCH_SIZE);
      auto expected = data.slice(0, BATCH_SIZE, BATCH_SIZE + OUTPUT_SIZE);
      auto loss = predictor.Train(input, expected);

      input = data.slice(0, OUTPUT_SIZE, BATCH_SIZE + OUTPUT_SIZE);
      auto prediction = predictor.Predict(input)[0].item<int64_t>();
      std::cout << "> " << prediction << " | Loss: " << loss << "\n";
      output_file << prediction << " ";

      prediction_x.push_back(epoch + 1);
      prediction_y.push_back(prediction);
      loss_x.push_back(epoch);
      loss_y.push_back(loss);

      dataset.erase(dataset.begin());
    }
  }

  input_file.close();
  output_file.close();

  PlotPredictions(expected_x, expected_y, prediction_x, prediction_y);
  PlotLoss(loss_x, loss_y);
}
