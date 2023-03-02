// sherpa-ncnn/csrc/zipformer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_ZIPFORMER_MODEL_H_
#define SHERPA_NCNN_CSRC_ZIPFORMER_MODEL_H_
#include <string>
#include <utility>
#include <vector>

#include "net.h"  // NOLINT
#include "sherpa-ncnn/csrc/model.h"

namespace sherpa_ncnn {
// Please refer to https://github.com/k2-fsa/icefall/pull/906
// for how the model is converted from icefall to ncnn
class ZipformerModel : public Model {
 public:
  explicit ZipformerModel(const ModelConfig &config);
#if __ANDROID_API__ >= 9
  ZipformerModel(AAssetManager *mgr, const ModelConfig &config);
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
    // pad_length 7, because the subsampling expression is
    // ((x_len - 7) // 2 + 1)//2, we need to pad 7 frames
    //
    // decode chunk length before subsample is 32 frames
    //
    // So each segment is pad_length + decode_chunk_length = 7 + 32 = 39
    return decode_chunk_length_ + pad_length_;
  }

  // Advance the feature extract by this number of frames after
  // running the encoder network
  int32_t Offset() const override { return decode_chunk_length_; }

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

  int32_t decode_chunk_length_ = 32;  // arg1, before subsampling
  int32_t num_left_chunks_ = 4;       // arg2
  int32_t pad_length_ = 7;            // arg3

  std::vector<int32_t> num_encoder_layers_;              // arg16
  std::vector<int32_t> encoder_dims_;                    // arg17
  std::vector<int32_t> attention_dims_;                  // arg18
  std::vector<int32_t> zipformer_downsampling_factors_;  // arg19
  std::vector<int32_t> cnn_module_kernels_;              // arg20

  std::vector<int32_t> encoder_input_indexes_;
  std::vector<int32_t> encoder_output_indexes_;

  std::vector<int32_t> decoder_input_indexes_;
  std::vector<int32_t> decoder_output_indexes_;

  std::vector<int32_t> joiner_input_indexes_;
  std::vector<int32_t> joiner_output_indexes_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_ZIPFORMER_MODEL_H_
