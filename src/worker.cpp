#include "worker.hpp"

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <filesystem>
#include <fstream>
#include <string>

#include "common/config.hpp"
#include "common/types.hpp"
#include "common/utils.hpp"
#include "torch/torch.h"

namespace fs = std::filesystem;

using std::chrono::duration;
using std::chrono::high_resolution_clock;

Worker::Worker(const std::string& key) : key_(key) {
#if VERBOSE
  figure_path_ = fs::path(FIGURE_PATH) / key;
  fs::create_directories(figure_path_);

  log_path_ = fs::path(LOG_PATH) / key;
  fs::create_directories(log_path_);

  log_file_.open(log_path_ / "output.log");
#endif
}

Worker::~Worker() {
#if VERBOSE
  /* Logging */

  log_file_ << "\nLoss: ";
  log_file_ << avg_train_loss_ << " (train avg)";
  log_file_ << " | " << avg_valid_loss_ << " (valid avg)";

  log_file_ << "\nTime: ";
  log_file_ << avg_time_ << " ms (avg)";
  log_file_ << " | " << max_time_ << " ms (max)\n";

  log_file_.close();

  if (expected_x_.size() > 0) {
    PlotPredictions(
        figure_path_ / "predictions.svg",
        expected_x_,
        expected_y_,
        prediction_x_,
        prediction_y_);
    PlotLoss(
        figure_path_ / "loss.svg",
        train_loss_x_,
        train_loss_y_,
        valid_loss_x_,
        valid_loss_y_);
  }
#endif
}

void Worker::Insert(const value_t& value) {
#if VERBOSE
  auto start_time = high_resolution_clock::now();
#endif

  dataset_.push_back(value);
  // Use naive prediction as a fallback
  prediction_ = value;

  auto size = dataset_.size();
  if (size < WINDOW_SIZE) return;

  /* Training */

  // Normalize data
  auto data = torch::tensor(dataset_) / MAX_SIZE;
  // Maintain a sliding window
  dataset_.erase(dataset_.begin());

  auto train_loss = predictor_.Train(data);

  /* Validation */

  auto valid_input = data.slice(0, size - SEQ_LEN, size);
  auto predictions = predictor_.Predict(valid_input);
  // Denormalize prediction
  prediction_ = predictions[0].item<double>() * MAX_SIZE;

  predictor_.UpdateLR();

#if VERBOSE
  /* Plotting */

  // Waiting for an acceptable training loss
  if (train_loss >= 0 && train_loss < LOSS_THRESHOLD) {
    start_plotting_ = true;
  }

  if (start_plotting_ && train_loss >= 0) {
    // Denormalize training loss
    train_loss *= MAX_SIZE;
    total_train_loss_ += train_loss;
    ++train_loss_count_;
    avg_train_loss_ = total_train_loss_ / train_loss_count_;

    if (epoch_ % PLOT_STEP == 0) {
      train_loss_x_.push_back(epoch_);
      train_loss_y_.push_back(avg_train_loss_);
    }
  }

  auto valid_loss = -1.0;
  if (prediction_y_.size() >= OUTPUT_SIZE) {
    // Normalize predictions
    auto prev_preds = torch::tensor(std::vector<value_t>{
                          prediction_y_.end() - OUTPUT_SIZE,
                          prediction_y_.end(),
                      }) /
                      MAX_SIZE;
    auto prev_expected = data.slice(0, size - OUTPUT_SIZE, size);
    valid_loss = predictor_.Loss(prev_preds, prev_expected).item<double>();
  }

  if (start_plotting_) {
    // Add 1 for it's a point in the future
    prediction_x_.push_back(epoch_ + 1);
    prediction_y_.push_back(prediction_);
  }
  expected_x_.push_back(epoch_);
  expected_y_.push_back(value);

  if (start_plotting_ && valid_loss >= 0) {
    // Denormalize validation loss
    valid_loss *= MAX_SIZE;
    total_valid_loss_ += valid_loss;
    ++valid_loss_count_;
    avg_valid_loss_ = total_valid_loss_ / valid_loss_count_;

    auto pred_epoch = epoch_ + 1 - OUTPUT_SIZE;
    if (pred_epoch % PLOT_STEP == 0) {
      valid_loss_x_.push_back(pred_epoch);
      valid_loss_y_.push_back(avg_valid_loss_);
    }
  }

  /* Logging */

  auto end_time = high_resolution_clock::now();
  auto time = duration<double, std::milli>(end_time - start_time).count();
  total_time_ += time;
  ++time_count_;
  avg_time_ = total_time_ / time_count_;
  max_time_ = std::max(max_time_, time);

  log_file_ << "#" << std::setw(6) << std::left << epoch_ + 1;
  log_file_ << " > " << prediction_;
  ++epoch_;

  log_file_ << " \tLoss: " << std::fixed << std::setprecision(5) << std::right;
  if (avg_train_loss_ >= 0) {
    log_file_ << std::setw(8) << avg_train_loss_ << " (train avg)";
  } else {
    log_file_ << std::string(20, ' ');
  }
  if (avg_valid_loss_ >= 0) {
    log_file_ << " | " << std::setw(8) << avg_valid_loss_ << " (valid avg)";
  } else {
    log_file_ << std::string(23, ' ');
  }

  log_file_ << " \tTime: " << time << " ms\n";
#endif
}
