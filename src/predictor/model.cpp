#include "model.hpp"

#include <torch/torch.h>

#include <tuple>

namespace nn = torch::nn;

using torch::Tensor;

GruNet::GruNet(
    int64_t input_size,
    int64_t hidden_size,
    int64_t output_size,
    int64_t n_layers,
    double dropout)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      output_size_(output_size),
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

std::tuple<Tensor, Tensor> GruNet::Forward(const Tensor& input, Tensor hidden) {
  auto [output, new_hidden] = gru_(input, hidden);
  output = fc_(relu_(output));
  return {output, new_hidden};
}

Tensor GruNet::InitHidden(int64_t batch_size) {
  auto weight = parameters().at(0).data();
  auto hidden = weight.new_zeros({n_layers_, batch_size, hidden_size_});
  return hidden;
}
