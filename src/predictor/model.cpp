#include "model.hpp"

#include <torch/torch.h>

#include <string>

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
               .bidirectional(true)
               .dropout(dropout)),
      fc_(nn::LinearOptions(hidden_size << 1, output_size)),
      relu_(),
      dropout_(nn::DropoutOptions(dropout)) {
  register_module("gru_", gru_);
  register_module("fc_", fc_);
  register_module("relu_", relu_);
  register_module("dropout_", dropout_);

  // Initialize weights
  auto params = gru_->named_parameters();
  for (auto&& param : params) {
    auto name = param.key();
    auto data = param.value();
    if (name.find("weight_ih") != std::string::npos) {
      nn::init::xavier_uniform_(data);
    } else if (name.find("weight_hh") != std::string::npos) {
      nn::init::orthogonal_(data);
    }
  }
}

Tensor GruNet::Forward(Tensor input) {
  input = input.reshape({batch_size_, 1, -1});
  auto [output, hidden_n] = gru_(input, hidden_);
  hidden_ = hidden_n;
  output = output.reshape({batch_size_, -1});
  output = fc_(output);
  return output.mean(0);
}

void GruNet::InitHidden(int64_t batch_size) {
  auto weight = parameters().at(0).data();
  hidden_ = weight.new_zeros({n_layers_ << 1, batch_size, hidden_size_});
}
