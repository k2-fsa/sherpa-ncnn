// sherpa-ncnn/csrc/conv-emformer-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa-ncnn/csrc/conv-emformer-model.h"

#include <regex>

#include "net.h"  // NOLINT

namespace sherpa_ncnn {

ConvEmformerModel::ConvEmformerModel(const ModelConfig &config)
    : num_threads_(config.num_threads) {
  InitEncoder(config.encoder_param, config.encoder_bin);
  InitDecoder(config.decoder_param, config.decoder_bin);
  InitJoiner(config.joiner_param, config.joiner_bin);

  InitEncoderInputOutputIndexes();
  InitDecoderInputOutputIndexes();
  InitJoinerInputOutputIndexes();
}

std::pair<ncnn::Mat, std::vector<ncnn::Mat>> ConvEmformerModel::RunEncoder(
    ncnn::Mat &features, const std::vector<ncnn::Mat> &states) {
  std::vector<ncnn::Mat> _states;

  const ncnn::Mat *p;
  if (states.empty()) {
    _states = GetEncoderInitStates();
    p = _states.data();
  } else {
    p = states.data();
  }

  ncnn::Extractor encoder_ex = encoder_.create_extractor();
  encoder_ex.set_num_threads(num_threads_);

  // Note: We ignore error check there
  encoder_ex.input(encoder_input_indexes_[0], features);
  for (int32_t i = 1; i != encoder_input_indexes_.size(); ++i) {
    encoder_ex.input(encoder_input_indexes_[i], p[i - 1]);
  }

  ncnn::Mat encoder_out;
  encoder_ex.extract(encoder_output_indexes_[0], encoder_out);

  std::vector<ncnn::Mat> next_states(num_layers_ * 4);
  for (int32_t i = 1; i != encoder_output_indexes_.size(); ++i) {
    encoder_ex.extract(encoder_output_indexes_[i], next_states[i - 1]);
  }

  return {encoder_out, next_states};
}

ncnn::Mat ConvEmformerModel::RunDecoder(ncnn::Mat &decoder_input) {
  ncnn::Extractor decoder_ex = decoder_.create_extractor();
  decoder_ex.set_num_threads(num_threads_);

  ncnn::Mat decoder_out;
  decoder_ex.input(decoder_input_indexes_[0], decoder_input);
  decoder_ex.extract(decoder_output_indexes_[0], decoder_out);
  decoder_out = decoder_out.reshape(decoder_out.w);

  return decoder_out;
}

ncnn::Mat ConvEmformerModel::RunJoiner(ncnn::Mat &encoder_out,
                                       ncnn::Mat &decoder_out) {
  auto joiner_ex = joiner_.create_extractor();
  joiner_ex.set_num_threads(num_threads_);
  joiner_ex.input(joiner_input_indexes_[0], encoder_out);
  joiner_ex.input(joiner_input_indexes_[1], decoder_out);

  ncnn::Mat joiner_out;
  joiner_ex.extract("out0", joiner_out);
  return joiner_out;
}

void ConvEmformerModel::InitEncoder(const std::string &encoder_param,
                                    const std::string &encoder_bin) {
  InitNet(encoder_, encoder_param, encoder_bin);
}

void ConvEmformerModel::InitDecoder(const std::string &decoder_param,
                                    const std::string &decoder_bin) {
  InitNet(decoder_, decoder_param, decoder_bin);
}

void ConvEmformerModel::InitJoiner(const std::string &joiner_param,
                                   const std::string &joiner_bin) {
  InitNet(joiner_, joiner_param, joiner_bin);
}

std::vector<ncnn::Mat> ConvEmformerModel::GetEncoderInitStates() const {
  std::vector<ncnn::Mat> states;
  states.reserve(num_layers_ * 4);

  for (int32_t i = 0; i != num_layers_; ++i) {
    auto s0 = ncnn::Mat(d_model_, memory_size_);
    auto s1 = ncnn::Mat(d_model_, left_context_length_);
    auto s2 = ncnn::Mat(d_model_, left_context_length_);
    auto s3 = ncnn::Mat(cnn_module_kernel_ - 1, d_model_);

    s0.fill(0);
    s1.fill(0);
    s2.fill(0);
    s3.fill(0);

    states.push_back(s0);
    states.push_back(s1);
    states.push_back(s2);
    states.push_back(s3);
  }

  return states;
}

void ConvEmformerModel::InitEncoderInputOutputIndexes() {
  // input indexes map
  // [0] -> in0, features,
  // [1] -> in1, layer0, s0
  // [2] -> in2, layer0, s1
  // [3] -> in3, layer0, s2
  // [4] -> in4, layer0, s3
  //
  // [5] -> in5, layer1, s0
  // [6] -> in6, layer1, s1
  // [7] -> in7, layer1, s2
  // [8] -> in8, layer1, s3
  //
  // until layer 11
  encoder_input_indexes_.resize(1 + num_layers_ * 4);

  // output indexes map
  // [0] -> out0, encoder_out
  //
  // [1] -> out1, layer0, s0
  // [2] -> out2, layer0, s1
  // [3] -> out3, layer0, s2
  // [4] -> out4, layer0, s3
  //
  // [5] -> out5, layer1, s0
  // [6] -> out6, layer1, s1
  // [7] -> out7, layer1, s2
  // [8] -> out8, layer1, s3
  encoder_output_indexes_.resize(1 + num_layers_ * 4);
  const auto &blobs = encoder_.blobs();

  std::regex in_regex("in(\\d+)");
  std::regex out_regex("out(\\d+)");

  std::smatch match;
  for (int32_t i = 0; i != blobs.size(); ++i) {
    const auto &b = blobs[i];
    if (std::regex_match(b.name, match, in_regex)) {
      auto index = std::atoi(match[1].str().c_str());
      encoder_input_indexes_[index] = i;
    } else if (std::regex_match(b.name, match, out_regex)) {
      auto index = std::atoi(match[1].str().c_str());
      encoder_output_indexes_[index] = i;
    }
  }
}

void ConvEmformerModel::InitDecoderInputOutputIndexes() {
  // input indexes map
  // [0] -> in0, decoder_input,
  decoder_input_indexes_.resize(1);

  // output indexes map
  // [0] -> out0, decoder_out,
  decoder_output_indexes_.resize(1);

  const auto &blobs = decoder_.blobs();
  for (int32_t i = 0; i != blobs.size(); ++i) {
    const auto &b = blobs[i];
    if (b.name == "in0") decoder_input_indexes_[0] = i;
    if (b.name == "out0") decoder_output_indexes_[0] = i;
  }
}

void ConvEmformerModel::InitJoinerInputOutputIndexes() {
  // input indexes map
  // [0] -> in0, encoder_input,
  // [1] -> in1, decoder_input,
  joiner_input_indexes_.resize(2);

  // output indexes map
  // [0] -> out0, joiner_out,
  joiner_output_indexes_.resize(1);

  const auto &blobs = joiner_.blobs();
  for (int32_t i = 0; i != blobs.size(); ++i) {
    const auto &b = blobs[i];
    if (b.name == "in0") joiner_input_indexes_[0] = i;
    if (b.name == "in1") joiner_input_indexes_[1] = i;
    if (b.name == "out0") joiner_output_indexes_[0] = i;
  }
}

}  // namespace sherpa_ncnn
