// sherpa-ncnn/csrc/conv-emformer-model.h
//
// Copyright (c)  2022  Xiaomi Corporation

#include "net.h"  // NOLINT
#include "sherpa-ncnn/csrc/model.h"

namespace sherpa_ncnn {
// Please refer to https://github.com/k2-fsa/icefall/pull/717
// for how the model is converted from icefall to ncnn
class ConvEmformerModel : public Model {
 public:
  explicit ConvEmformerModel(const ModelConfig &config);

  std::pair<ncnn::Mat, std::vector<ncnn::Mat>> RunEncoder(
      ncnn::Mat &features, const std::vector<ncnn::Mat> &states) override;

  ncnn::Mat RunDecoder(ncnn::Mat &decoder_input) override;

  ncnn::Mat RunJoiner(ncnn::Mat &encoder_out, ncnn::Mat &decoder_out) override;

  int32_t Segment() const override {
    // chunk_length 32
    // right_context 8
    // subsampling factor 4
    //
    // segment = 32 + (8 + 2 * 4 + 3) = 32 + 19 = 51
    return 51;
  }

  // Advance the feature extract by this number of frames after
  // running the encoder network
  int32_t Offset() const override { return chunk_length_; }

 private:
  void InitEncoder(const std::string &encoder_param,
                   const std::string &encoder_bin);
  void InitDecoder(const std::string &decoder_param,
                   const std::string &decoder_bin);
  void InitJoiner(const std::string &joiner_param,
                  const std::string &joiner_bin);

  void InitStateNames();

  std::vector<ncnn::Mat> GetEncoderInitStates() const;

 private:
  ncnn::Net encoder_;
  ncnn::Net decoder_;
  ncnn::Net joiner_;

  int32_t num_threads_;

  int32_t num_layers_ = 12;
  int32_t memory_size_ = 32;
  int32_t cnn_module_kernel_ = 31;
  int32_t left_context_length_ = 32 / 4;
  int32_t chunk_length_ = 32;
  int32_t right_context_length_ = 8;
  int32_t d_model_ = 512;

  std::vector<std::string> in_state_names_;
  std::vector<std::string> out_state_names_;
};

}  // namespace sherpa_ncnn
