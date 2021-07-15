#ifndef SRC_PREDICTOR_MODEL_HPP_
#define SRC_PREDICTOR_MODEL_HPP_

#include <torch/torch.h>

class GruNet : public torch::nn::Module {
 public:
  GruNet(
      int64_t input_size,
      int64_t hidden_size,
      int64_t output_size,
      int64_t batch_size,
      int64_t n_layers,
      double dropout = 0.2);

  torch::Tensor Forward(torch::Tensor input);
  void InitHidden(int64_t batch_size);

 private:
  int64_t hidden_size_;
  int64_t batch_size_;
  int64_t n_layers_;

  torch::nn::GRU gru_;
  torch::nn::Linear fc_;
  torch::nn::ReLU relu_;

  torch::Tensor hidden_;
};

#endif  // SRC_PREDICTOR_MODEL_HPP_
