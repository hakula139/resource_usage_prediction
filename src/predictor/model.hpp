#ifndef SRC_PREDICTOR_MODEL_HPP_
#define SRC_PREDICTOR_MODEL_HPP_

#include "torch/torch.h"

class GruNet : public torch::nn::Module {
 public:
  /**
   * Args:
   *   `hidden_size`: the dimension of the hidden state
   *   `seq_len`: the size of each batch data (sequence length)
   *   `n_layers`: the depth of recurrent layers
   *   `dropout`: the dropout rate of each recurrent layer
   */
  GruNet(
      int64_t hidden_size,
      int64_t seq_len,
      int64_t n_layers = 1,
      double dropout = 0.0);

  /**
   * Args:
   *   `input`: shape(batch_size, seq_len)
   *
   * Returns:
   *   `output`: shape(batch_size, seq_len, hidden_size)
   *   `hidden`: shape(batch_size, hidden_size)
   */
  torch::Tensor Forward(torch::Tensor input);

  /**
   * Initialize hidden state.
   *
   *   Args:
   *     `batch_size`: batch size
   *
   *   Returns:
   *     shape(n_layers, batch_size, hidden_size)
   */
  torch::Tensor InitHidden(int64_t batch_size);

  friend class Predictor;

 private:
  int64_t hidden_size_;
  int64_t seq_len_;
  int64_t n_layers_;

  torch::nn::GRU encoder_;
  torch::nn::GRUCell decoder_;
  torch::nn::Linear fc_;
  torch::nn::ReLU relu_;
  torch::nn::Dropout dropout_;
};

#endif  // SRC_PREDICTOR_MODEL_HPP_
