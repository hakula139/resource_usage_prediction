#include "model.hpp"

#include <torch/torch.h>

namespace nn = torch::nn;

using torch::Tensor;

GruNet::GruNet(
    int64_t input_size,
    int64_t hidden_size,
    int64_t output_size,
    int64_t batch_size,
    int64_t n_layers,
    double dropout)
    : hidden_size_(hidden_size),
      batch_size_(batch_size),
      n_layers_(n_layers),
      gru_(nn::GRUOptions(input_size, hidden_size)
               .num_layers(n_layers)
               .batch_first(true)
               .dropout(dropout)),
      fc_(nn::LinearOptions(hidden_size, output_size)),
      relu_() {
  register_module("gru", gru_);
  register_module("fc", fc_);
  register_module("relu", relu_);
}

Tensor GruNet::Forward(Tensor input) {
  input = input.reshape({batch_size_, 1, -1});
  auto [output, hidden_n] = gru_(input, hidden_);
  hidden_ = hidden_n;
  output = output.reshape({batch_size_, -1});
  output = fc_(output);
  return output[batch_size_ - 1];
}

void GruNet::InitHidden(int64_t batch_size) {
  auto weight = parameters().at(0).data();
  hidden_ = weight.new_zeros({n_layers_, batch_size, hidden_size_});
}
