#ifndef SRC_PREDICTOR_PREDICTOR_HPP_
#define SRC_PREDICTOR_PREDICTOR_HPP_

#include "config.hpp"
#include "model.hpp"
#include "torch/torch.h"

class Predictor {
 public:
  Predictor()
      : model_(HIDDEN_SIZE, SEQ_LEN, N_LAYERS, DROPOUT),
        criterion_(),
        optimizer_(model_.parameters(), {LEARNING_RATE}),
        scheduler_(optimizer_, 1000, 0.8) {}

  double Train(const torch::Tensor& batch_data);
  torch::Tensor Predict(const torch::Tensor& batch_data);
  torch::Tensor Loss(const torch::Tensor& output, const torch::Tensor& target);
  void UpdateLR();

 private:
  GruNet model_;
  torch::nn::MSELoss criterion_;
  torch::optim::AdamW optimizer_;
  // TODO: Switch to ReduceLROnPlateau after PyTorch C++ frontend has
  // implemented this feature.
  torch::optim::StepLR scheduler_;
};

#endif  // SRC_PREDICTOR_PREDICTOR_HPP_
