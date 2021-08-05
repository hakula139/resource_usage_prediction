#include "predictor.hpp"

#include <iostream>
#include <vector>

#include "config.hpp"
#include "torch/torch.h"

namespace nn = torch::nn;
namespace optim = torch::optim;

using torch::Tensor;

double Predictor::Train(const Tensor& batch_data) {
  model_.train();
  model_.zero_grad();

  auto batch_size = batch_data.size(0) - SEQ_LEN - OUTPUT_SIZE;
  auto step = (WINDOW_SIZE / SEQ_LEN) << 1;
  int64_t train_size = batch_size / step + 1;

  auto train_data = torch::zeros({train_size, SEQ_LEN});
  auto expected = torch::zeros({train_size, OUTPUT_SIZE});
  for (auto i = batch_size; i >= 0; i -= step) {
    auto batch_i = i / step;
    train_data[batch_i] = batch_data.slice(0, i, i + SEQ_LEN);
    expected[batch_i] = batch_data.slice(
        0, i + SEQ_LEN, i + SEQ_LEN + OUTPUT_SIZE);
  }

  auto output = model_.Forward(train_data);
  auto loss = Loss(output, expected);
  auto cur_loss = loss.item<double>();

  loss.backward();
  optimizer_.step();
  return cur_loss;
}

Tensor Predictor::Predict(const Tensor& batch_data) {
  model_.eval();

  auto predict_data = batch_data.unsqueeze(0);

  torch::NoGradGuard no_grad;
  auto output = model_.Forward(predict_data);
  output = model_.relu_(output);
  return output;
}

Tensor Predictor::Loss(const Tensor& output, const Tensor& target) {
  return criterion_(output, target).sqrt();
}

void Predictor::UpdateLR() {
  auto options = static_cast<optim::AdamWOptions&>(optimizer_.defaults());
  auto learning_rate = options.lr();
  if (learning_rate > 1e-3) scheduler_.step();
}
