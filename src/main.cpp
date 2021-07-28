#include <torch/torch.h>

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
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

  std::cout << "Server started.\n" << std::setprecision(5);

  std::ifstream input_file(INPUT_PATH);
  std::ofstream output_file(OUTPUT_PATH);

  // Points for plotting
  std::vector<int64_t> expected_x, prediction_x;
  std::vector<int64_t> expected_y, prediction_y;
  std::vector<int64_t> train_loss_x, naive_loss_x;
  std::vector<double> train_loss_y, naive_loss_y;

  auto start_plotting = false;

  double total_time = 0.0, max_time = 0.0;
  int64_t time_count = 0;
  double total_loss = 0.0, avg_loss = -1.0;
  int64_t loss_count = 0;

  for (int64_t epoch = 1; !input_file.eof(); ++epoch) {
    int64_t cur_data = -1;
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

      if (train_loss < LOSS_THRESHOLD) {
        start_plotting = true;
      } else {
        start_plotting = false;
        train_loss_x.clear();
        train_loss_y.clear();
        total_loss = 0;
        loss_count = 0;
      }

      if (start_plotting) {
        train_loss_x.push_back(epoch);
        train_loss_y.push_back(train_loss);
        total_loss += train_loss;
        ++loss_count;
      }

      avg_loss = loss_count > 0 ? total_loss / loss_count : -1.0;
      // predictor.UpdateLR();

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

      std::cout << "#" << std::setw(6) << std::left << epoch + 1;
      std::cout << " > " << prediction << " (naive: " << naive_pred << ")  ";
      std::cout << " \tLoss:" << std::right;
      std::cout << " " << std::setw(9) << train_loss << " (current)";
      std::cout << " | " << std::setw(3) << naive_loss << " (naive)";
      if (avg_loss >= 0) {
        std::cout << " | " << std::setw(9) << avg_loss << " (average)";
      } else {
        std::cout << std::string(22, ' ');
      }
      std::cout << " \tTime: " << time << " ms" << std::endl;
      output_file << prediction << " ";
    }
  }

  input_file.close();
  output_file.close();

  auto total_naive_loss = std::accumulate(
      naive_loss_y.begin(), naive_loss_y.end(), 0.0);
  auto avg_naive_loss = total_naive_loss / naive_loss_y.size();
  auto avg_time = total_time / time_count;

  std::cout << "Loss:";
  std::cout << " " << avg_loss << " (average)";
  std::cout << " | " << avg_naive_loss << " (naive)\n";
  std::cout << "Time:";
  std::cout << " " << avg_time << " ms (average)";
  std::cout << " | " << max_time << " ms (max)" << std::endl;

  PlotPredictions(expected_x, expected_y, prediction_x, prediction_y);
  PlotTrainLoss(train_loss_x, train_loss_y, naive_loss_x, naive_loss_y);
}
