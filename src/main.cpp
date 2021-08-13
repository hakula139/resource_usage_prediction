#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "common/config.hpp"
#include "common/utils.hpp"
#include "predictor/predictor.hpp"
#include "torch/torch.h"

namespace fs = std::filesystem;

using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main() {
  std::vector<int64_t> dataset;
  Predictor predictor;

  fs::create_directories(FIGURE_DIR);

  std::cout << "Server started.\n";

  std::ifstream input_file(INPUT_PATH);
  std::ofstream output_file(OUTPUT_PATH);

  // Points for plotting
  std::vector<int64_t> expected_x, prediction_x;
  std::vector<int64_t> expected_y, prediction_y;
  std::vector<int64_t> train_loss_x, valid_loss_x;
  std::vector<double> train_loss_y, valid_loss_y;

  double total_time = 0.0, max_time = 0.0;
  int64_t time_count = 0;
  double total_train_loss = 0.0, avg_train_loss = -1.0;
  double total_valid_loss = 0.0, avg_valid_loss = -1.0;
  int64_t train_loss_count = 0, valid_loss_count = 0;

  auto start_plotting = false;

  for (int64_t epoch = 1; !input_file.eof(); ++epoch) {
    int64_t cur_data = -1;
    input_file >> cur_data;
    if (cur_data < 0) break;

    auto start_time = high_resolution_clock::now();

    expected_x.push_back(epoch);
    expected_y.push_back(cur_data);

    dataset.push_back(cur_data);

    auto size = dataset.size();
    if (size < WINDOW_SIZE) continue;

    /* Training */

    auto data = torch::tensor(dataset) / MAX_SIZE;
    dataset.erase(dataset.begin());

    auto train_loss = predictor.Train(data);

    // Waiting for an acceptable training loss
    if (train_loss >= 0 && train_loss < LOSS_THRESHOLD) {
      start_plotting = true;
    }

    if (start_plotting && train_loss >= 0) {
      train_loss *= MAX_SIZE;
      total_train_loss += train_loss;
      ++train_loss_count;
      avg_train_loss = total_train_loss / train_loss_count;

      if (epoch % PLOT_STEP == 0) {
        train_loss_x.push_back(epoch);
        train_loss_y.push_back(avg_train_loss);
      }
    }

    /* Validation */

    auto valid_input = data.slice(0, size - SEQ_LEN, size);
    auto predictions = predictor.Predict(valid_input);

    auto valid_loss = -1.0;
    if (prediction_y.size() >= OUTPUT_SIZE) {
      auto prev_preds = torch::tensor(std::vector<int64_t>{
                            prediction_y.end() - OUTPUT_SIZE,
                            prediction_y.end(),
                        }) /
                        MAX_SIZE;
      auto prev_expected = data.slice(0, size - OUTPUT_SIZE, size);
      valid_loss = predictor.Loss(prev_preds, prev_expected).item<double>();
    }

    int64_t prediction = round(predictions[0].item<double>() * MAX_SIZE);

    if (start_plotting) {
      prediction_x.push_back(epoch + 1);
      prediction_y.push_back(prediction);
    }

    if (start_plotting && valid_loss >= 0) {
      valid_loss *= MAX_SIZE;
      total_valid_loss += valid_loss;
      ++valid_loss_count;
      avg_valid_loss = total_valid_loss / valid_loss_count;

      auto pred_epoch = epoch + 1 - OUTPUT_SIZE;
      if (pred_epoch % PLOT_STEP == 0) {
        valid_loss_x.push_back(pred_epoch);
        valid_loss_y.push_back(avg_valid_loss);
      }
    }

    predictor.UpdateLR();

    auto end_time = high_resolution_clock::now();
    auto time = duration<double, std::milli>(end_time - start_time).count();
    total_time += time;
    max_time = std::max(max_time, time);
    ++time_count;

    std::cout << "#" << std::setw(6) << std::left << epoch + 1;
    std::cout << " > " << prediction;

    std::cout << " \tLoss: " << std::fixed << std::setprecision(5)
              << std::right;
    if (avg_train_loss >= 0) {
      std::cout << std::setw(8) << avg_train_loss << " (train avg)";
    } else {
      std::cout << std::string(20, ' ');
    }
    if (avg_valid_loss >= 0) {
      std::cout << " | " << std::setw(8) << avg_valid_loss << " (valid avg)";
    } else {
      std::cout << std::string(23, ' ');
    }

    std::cout << " \tTime: " << time << " ms" << std::endl;

    output_file << prediction << " ";

    // Sleep for a while before reading next value
    std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_TIME));
  }

  input_file.close();
  output_file.close();

  auto avg_time = total_time / time_count;
  std::cout << "Loss: ";
  std::cout << avg_valid_loss << " (valid avg)\n";
  std::cout << "Time: ";
  std::cout << avg_time << " ms (avg)";
  std::cout << " | " << max_time << " ms (max)" << std::endl;

  PlotPredictions(expected_x, expected_y, prediction_x, prediction_y);
  PlotLoss(train_loss_x, train_loss_y, valid_loss_x, valid_loss_y);
}
