#include "predictor.hpp"

#include <iostream>

#include "config.hpp"
#include "torch/torch.h"

namespace nn = torch::nn;

using torch::Tensor;

double Predictor::Train(const Tensor& batch_data, const Tensor& expected) {
  model_.train();
  model_.zero_grad();
  model_.InitHidden(BATCH_SIZE);

  auto output = model_.Forward(batch_data / MAX_SIZE) * MAX_SIZE;

  auto loss = criterion_(output, expected.to(torch::kFloat));
  auto cur_loss = loss.item<double>();

  loss.backward();
  optimizer_.step();
  return cur_loss;
}

Tensor Predictor::Predict(const Tensor& batch_data) {
  model_.eval();
  model_.InitHidden(BATCH_SIZE);

  auto output = model_.Forward(batch_data / MAX_SIZE) * MAX_SIZE;
  return model_.relu_(output);
}
