#include <torch/torch.h>

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "common/config.hpp"
#include "common/utils.hpp"
#include "predictor/predictor.hpp"

namespace fs = std::filesystem;

using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main() {
  std::vector<int64_t> dataset;
  Predictor predictor;

  fs::create_directories(FIGURE_DIR);

  std::ifstream input_file(INPUT_PATH);
  std::ofstream output_file(OUTPUT_PATH);

  // Points for plotting
  std::vector<int64_t> expected_x, prediction_x;
  std::vector<int64_t> expected_y, prediction_y;
  std::vector<int64_t> train_loss_x, naive_loss_x;
  std::vector<double> train_loss_y, naive_loss_y;

  auto total_time = 0.0;
  auto max_time = 0.0;
  auto time_count = 0;

  std::cout << "Server started.\n" << std::setprecision(5);

  for (int64_t epoch = 1; !input_file.eof(); ++epoch) {
    int64_t cur_data;
    input_file >> cur_data;
    if (cur_data < 0) break;

    auto start_time = high_resolution_clock::now();
    dataset.push_back(cur_data);

    expected_x.push_back(epoch);
    expected_y.push_back(cur_data);

    auto size = dataset.size();
    if (size == BATCH_SIZE + OUTPUT_SIZE) {
      auto data = torch::tensor(dataset).to(torch::kFloat);
      dataset.erase(dataset.begin());

      auto train_input = data.slice(0, 0, BATCH_SIZE);
      auto expected = data.slice(0, BATCH_SIZE, size);
      auto train_loss = predictor.Train(train_input, expected);
      train_loss_x.push_back(epoch);
      train_loss_y.push_back(train_loss);

      auto naive_preds = data.slice(0, BATCH_SIZE - OUTPUT_SIZE, BATCH_SIZE);
      auto naive_loss = predictor.Loss(naive_preds, expected).item<double>();
      naive_loss_x.push_back(epoch);
      naive_loss_y.push_back(naive_loss);

      auto valid_input = data.slice(0, OUTPUT_SIZE, size);
      auto predictions = predictor.Predict(valid_input);

      auto prediction = round(predictions[0].item<double>());
      auto naive_pred = round(naive_preds[0].item<double>());
      prediction_x.push_back(epoch + 1);
      prediction_y.push_back(prediction);

      auto end_time = high_resolution_clock::now();
      auto time = duration<double, std::milli>(end_time - start_time).count();
      total_time += time;
      max_time = std::max(max_time, time);
      ++time_count;

      std::cout << "> " << prediction << " (naive: " << naive_pred << ") \t";
      std::cout << "Loss: ";
      std::cout << std::setw(10) << train_loss << " (train) | ";
      std::cout << naive_loss << " (naive) \t";
      std::cout << "Time: " << time << " ms\n";
      output_file << prediction << " ";
    }
  }

  input_file.close();
  output_file.close();

  std::cout << "Time: ";
  std::cout << total_time / time_count << " ms (average) | ";
  std::cout << max_time << " ms (max)\n";

  PlotPredictions(expected_x, expected_y, prediction_x, prediction_y);
  PlotTrainLoss(train_loss_x, train_loss_y, naive_loss_x, naive_loss_y);
}
