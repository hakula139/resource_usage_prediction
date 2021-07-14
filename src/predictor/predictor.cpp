#include "predictor.hpp"

#include <iostream>

#include "torch/torch.h"

namespace nn = torch::nn;

using torch::Tensor;

double Predictor::Train(const Tensor& batch_data, const Tensor& expected) {
  model_.train();
  auto hidden = model_.InitHidden(batch_data.size(0));
  model_.zero_grad();

  std::cout << batch_data << "\n" << hidden << "\n";

  auto [output, new_hidden] = model_.Forward(batch_data, hidden);
  hidden = new_hidden;

  std::cout << output << "\n" << new_hidden << "\n" << expected << "\n";

  auto loss = criterion_(output, expected);
  auto cur_loss = loss.item<double>();

  loss.backward();
  optimizer_.step();
  return cur_loss;
}

double Predictor::Predict(const Tensor& batch_data) {
  model_.eval();
  auto hidden = model_.InitHidden(SEQ_LEN);

  auto [output, new_hidden] = model_.Forward(batch_data, hidden);
  hidden = new_hidden;

  auto prediction = output[0].item<double>();
  return prediction;
}
