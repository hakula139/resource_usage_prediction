#include "model.hpp"

#include <string>

#include "torch/torch.h"

namespace nn = torch::nn;

using torch::Tensor;

GruNet::GruNet(
    int64_t hidden_size, int64_t seq_len, int64_t n_layers, double dropout)
    : hidden_size_(hidden_size),
      seq_len_(seq_len),
      n_layers_(n_layers),
      encoder_(nn::GRUOptions(1, hidden_size)
                   .num_layers(n_layers)
                   .batch_first(true)
                   .dropout(dropout)),
      decoder_(nn::GRUCellOptions(1, hidden_size)),
      fc_(nn::LinearOptions(hidden_size, 1)),
      relu_(),
      dropout_(nn::DropoutOptions(dropout)) {
  register_module("encoder_", encoder_);
  register_module("fc_", fc_);
  register_module("relu_", relu_);
  register_module("dropout_", dropout_);

  // Initialize weights
  auto params = encoder_->named_parameters();
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
  auto batch_size = input.size(0);
  auto init_hidden = InitHidden(batch_size);

  /**
   * `enc_output`: shape(batch_size, seq_len, hidden_size)
   * `enc_hidden`: shape(batch_size, hidden_size)
   */
  auto [enc_output, enc_hidden] = encoder_(input.unsqueeze(2), init_hidden);
  // Only use the last layer in hidden state
  enc_hidden = enc_hidden[n_layers_ - 1];

  /**
   * `dec_output`: shape(batch_size, 1)
   * `dec_hidden`: shape(batch_size, hidden_size)
   */
  // Only use the last value in each sequence
  auto last_input = input.index({torch::indexing::Slice(), seq_len_ - 1});
  auto dec_hidden = decoder_(last_input.unsqueeze(1), enc_hidden);
  auto dec_output = fc_(dec_hidden);
  dec_hidden = dropout_(dec_hidden);

  return dec_output;
}

Tensor GruNet::InitHidden(int64_t batch_size) {
  auto weight = parameters().at(0).data();
  return weight.new_zeros({n_layers_, batch_size, hidden_size_});
}
