#ifndef SRC_PREDICTOR_PREDICTOR_HPP_
#define SRC_PREDICTOR_PREDICTOR_HPP_

#include <torch/torch.h>

#include "config.hpp"
#include "model.hpp"

class Predictor {
 public:
  Predictor()
      : model_(
            INPUT_SIZE,
            HIDDEN_SIZE,
            OUTPUT_SIZE,
            BATCH_SIZE,
            N_LAYERS,
            DROPOUT),
        criterion_(),
        optimizer_(model_.parameters(), {LEARNING_RATE}) {}

  double Train(const torch::Tensor& batch_data, const torch::Tensor& expected);
  torch::Tensor Predict(const torch::Tensor& batch_data);
  torch::Tensor Loss(const torch::Tensor& output, const torch::Tensor& target);

 private:
  GruNet model_;
  torch::nn::MSELoss criterion_;
  torch::optim::Adam optimizer_;
};

#endif  // SRC_PREDICTOR_PREDICTOR_HPP_
