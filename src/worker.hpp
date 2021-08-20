#ifndef SRC_WORKER_HPP_
#define SRC_WORKER_HPP_

#include <filesystem>
#include <mutex>  // NOLINT(build/c++11)
#include <string>
#include <vector>

#include "common/types.hpp"
#include "predictor/predictor.hpp"

class Worker {
 public:
  explicit Worker(const std::string& key);
  ~Worker();

  void Insert(value_t value);
  value_t prediction() const { return prediction_; }

  void Lock() { mutex_.lock(); }
  void Unlock() { mutex_.unlock(); }

 private:
  const std::string key_;
  std::mutex mutex_;

  std::vector<value_t> dataset_;
  Predictor predictor_;
  value_t prediction_ = 0;

#if VERBOSE

  // Plotting related

  std::vector<int64_t> expected_x_, prediction_x_;
  std::vector<value_t> expected_y_, prediction_y_;
  std::vector<int64_t> train_loss_x_, valid_loss_x_;
  std::vector<double> train_loss_y_, valid_loss_y_;

  int64_t epoch_ = 1;
  bool start_plotting_ = false;
  std::filesystem::path figure_path_;

  // Statistics

  double total_time_ = 0.0;
  int64_t time_count_ = 0;
  double avg_time_ = -1.0;
  double max_time_ = 0.0;

  double total_train_loss_ = 0.0;
  int64_t train_loss_count_ = 0;
  double avg_train_loss_ = -1.0;

  double total_valid_loss_ = 0.0;
  int64_t valid_loss_count_ = 0;
  double avg_valid_loss_ = -1.0;

  // Logging related

  std::filesystem::path log_path_;
  std::ofstream log_file_;
#endif
};

#endif  // SRC_WORKER_HPP_
