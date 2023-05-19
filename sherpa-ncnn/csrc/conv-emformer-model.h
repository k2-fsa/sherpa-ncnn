// sherpa-ncnn/csrc/conv-emformer-model.h
//
// Copyright (c)  2022  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_CONV_EMFORMER_MODEL_H_
#define SHERPA_NCNN_CSRC_CONV_EMFORMER_MODEL_H_
#include <string>
#include <utility>
#include <vector>

#include "net.h"  // NOLINT
#include "sherpa-ncnn/csrc/model.h"

namespace sherpa_ncnn {
// Please refer to https://github.com/k2-fsa/icefall/pull/717
// for how the model is converted from icefall to ncnn
class ConvEmformerModel : public Model {
 public:
  explicit ConvEmformerModel(const ModelConfig &config);
#if __ANDROID_API__ >= 9
  ConvEmformerModel(AAssetManager *mgr, const ModelConfig &config);
#endif

  ncnn::Net &GetEncoder() override { return encoder_; }
  ncnn::Net &GetDecoder() override { return decoder_; }
  ncnn::Net &GetJoiner() override { return joiner_; }

  std::vector<ncnn::Mat> GetEncoderInitStates() const override;

  std::pair<ncnn::Mat, std::vector<ncnn::Mat>> RunEncoder(
      ncnn::Mat &features, const std::vector<ncnn::Mat> &states) override;

  std::pair<ncnn::Mat, std::vector<ncnn::Mat>> RunEncoder(
      ncnn::Mat &features, const std::vector<ncnn::Mat> &states,
      ncnn::Extractor *extractor) override;

  ncnn::Mat RunDecoder(ncnn::Mat &decoder_input) override;

  ncnn::Mat RunDecoder(ncnn::Mat &decoder_input,
                       ncnn::Extractor *extractor) override;

  ncnn::Mat RunJoiner(ncnn::Mat &encoder_out, ncnn::Mat &decoder_out) override;

  ncnn::Mat RunJoiner(ncnn::Mat &encoder_out, ncnn::Mat &decoder_out,
                      ncnn::Extractor *extractor) override;

  int32_t Segment() const override {
    // chunk_length 32
    // right_context 8
    // subsampling factor 4
    //
    // segment = 32 + (8 + 2 * 4 + 3) = 32 + 19 = 51
    return chunk_length_ + (right_context_length_ + 2 * 4 + 3);
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

  void InitEncoderPostProcessing();

#if __ANDROID_API__ >= 9
  void InitEncoder(AAssetManager *mgr, const std::string &encoder_param,
                   const std::string &encoder_bin);
  void InitDecoder(AAssetManager *mgr, const std::string &decoder_param,
                   const std::string &decoder_bin);
  void InitJoiner(AAssetManager *mgr, const std::string &joiner_param,
                  const std::string &joiner_bin);
#endif

  void InitEncoderInputOutputIndexes();
  void InitDecoderInputOutputIndexes();
  void InitJoinerInputOutputIndexes();

 private:
  ncnn::Net encoder_;
  ncnn::Net decoder_;
  ncnn::Net joiner_;

  int32_t num_layers_ = 12;               // arg1
  int32_t memory_size_ = 32;              // arg2
  int32_t cnn_module_kernel_ = 31;        // arg3
  int32_t left_context_length_ = 32 / 4;  // arg4
  int32_t chunk_length_ = 32;             // arg5
  int32_t right_context_length_ = 8;      // arg6
  int32_t d_model_ = 512;                 // arg7

  std::vector<int32_t> encoder_input_indexes_;
  std::vector<int32_t> encoder_output_indexes_;

  std::vector<int32_t> decoder_input_indexes_;
  std::vector<int32_t> decoder_output_indexes_;

  std::vector<int32_t> joiner_input_indexes_;
  std::vector<int32_t> joiner_output_indexes_;
};

}  // namespace sherpa_ncnn
#endif  // SHERPA_NCNN_CSRC_CONV_EMFORMER_MODEL_H_
