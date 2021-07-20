#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "common/config.hpp"
#include "common/utils.hpp"
#include "predictor/predictor.hpp"

namespace fs = std::filesystem;

int main() {
  std::vector<int64_t> dataset;
  Predictor predictor;

  fs::create_directories(FIGURE_DIR);

  std::ifstream input_file(INPUT_PATH);
  std::ofstream output_file(OUTPUT_PATH);

  // Points for plotting
  std::vector<int64_t> expected_x, prediction_x;
  std::vector<int64_t> expected_y, prediction_y;
  std::vector<int64_t> train_loss_x, valid_loss_x, naive_loss_x;
  std::vector<double> train_loss_y, valid_loss_y, naive_loss_y;

  std::cout << "Server started.\n";

  for (int64_t epoch = 1; !input_file.eof(); ++epoch) {
    int64_t cur_data;
    input_file >> cur_data;
    if (cur_data < 0) break;
    dataset.push_back(cur_data);

    expected_x.push_back(epoch);
    expected_y.push_back(cur_data);

    auto size = dataset.size();
    if (size == BATCH_SIZE + OUTPUT_SIZE) {
      auto data = torch::tensor(dataset).to(torch::kFloat);
      dataset.erase(dataset.begin());

      auto input = data.slice(0, 0, BATCH_SIZE);
      auto expected = data.slice(0, BATCH_SIZE, size);
      auto train_loss = predictor.Train(input, expected);
      train_loss_x.push_back(epoch);
      train_loss_y.push_back(train_loss);

      input = data.slice(0, OUTPUT_SIZE, BATCH_SIZE + OUTPUT_SIZE);
      auto predictions = predictor.Predict(input);
      auto valid_loss = predictor.Loss(predictions, expected).item<double>();
      valid_loss_x.push_back(epoch);
      valid_loss_y.push_back(valid_loss);

      auto naive_preds = data.slice(0, BATCH_SIZE - OUTPUT_SIZE, BATCH_SIZE);
      auto naive_loss = predictor.Loss(naive_preds, expected).item<double>();
      naive_loss_x.push_back(epoch);
      naive_loss_y.push_back(naive_loss);

      auto prediction = round(predictions[0].item<double>());
      auto naive_pred = round(naive_preds[0].item<double>());
      prediction_x.push_back(epoch + 1);
      prediction_y.push_back(prediction);

      std::cout << "> " << prediction << " (" << naive_pred << ") \t"
                << "Loss: " << train_loss << " (train) | " << valid_loss
                << " (valid) | " << naive_loss << " (naive)\n";
      output_file << prediction << " ";
    }
  }

  input_file.close();
  output_file.close();

  PlotPredictions(expected_x, expected_y, prediction_x, prediction_y);
  PlotTrainLoss(train_loss_x, train_loss_y, naive_loss_x, naive_loss_y);
  PlotValidLoss(valid_loss_x, valid_loss_y, naive_loss_x, naive_loss_y);
}
