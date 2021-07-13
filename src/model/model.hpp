#ifndef SRC_MODEL_MODEL_HPP_
#define SRC_MODEL_MODEL_HPP_

#include <torch/torch.h>

#include <tuple>

class GruNet : public torch::nn::Module {
 public:
  GruNet(
      int64_t input_size,
      int64_t hidden_size,
      int64_t output_size,
      int64_t n_layers,
      double dropout = 0.2);

  std::tuple<torch::Tensor, torch::Tensor> Forward(
      const torch::Tensor& input, torch::Tensor hidden);

  torch::Tensor InitHidden(int64_t batch_size);

 private:
  int64_t input_size_;
  int64_t hidden_size_;
  int64_t output_size_;
  int64_t n_layers_;

  torch::nn::GRU gru_;
  torch::nn::Linear fc_;
  torch::nn::ReLU relu_;
};

#endif  // SRC_MODEL_MODEL_HPP_
