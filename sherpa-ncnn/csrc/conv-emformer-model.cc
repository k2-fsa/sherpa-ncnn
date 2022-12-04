// sherpa-ncnn/csrc/conv-emformer-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "sherpa-ncnn/csrc/conv-emformer-model.h"

#include "net.h"  // NOLINT

namespace sherpa_ncnn {

ConvEmformerModel::ConvEmformerModel(const ModelConfig &config)
    : num_threads_(config.num_threads) {
  InitEncoder(config.encoder_param, config.encoder_bin);
  InitDecoder(config.decoder_param, config.decoder_bin);
  InitJoiner(config.joiner_param, config.joiner_bin);

  InitStateNames();
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
  encoder_ex.input("in0", features);
  for (int32_t i = 0; i != num_layers_; ++i) {
    int32_t n = i * 4;

    encoder_ex.input(in_state_names_[n].c_str(), p[n]);
    encoder_ex.input(in_state_names_[n + 1].c_str(), p[n + 1]);
    encoder_ex.input(in_state_names_[n + 2].c_str(), p[n + 2]);
    encoder_ex.input(in_state_names_[n + 3].c_str(), p[n + 3]);
  }

  ncnn::Mat encoder_out;
  encoder_ex.extract("out0", encoder_out);

  std::vector<ncnn::Mat> next_states(num_layers_ * 4);
  for (int32_t i = 0; i != num_layers_; ++i) {
    int32_t n = i * 4;
    encoder_ex.extract(out_state_names_[n].c_str(), next_states[n]);
    encoder_ex.extract(out_state_names_[n + 1].c_str(), next_states[n + 1]);
    encoder_ex.extract(out_state_names_[n + 2].c_str(), next_states[n + 2]);
    encoder_ex.extract(out_state_names_[n + 3].c_str(), next_states[n + 3]);
  }

  return {encoder_out, next_states};
}

ncnn::Mat ConvEmformerModel::RunDecoder(ncnn::Mat &decoder_input) {
  ncnn::Extractor decoder_ex = decoder_.create_extractor();
  decoder_ex.set_num_threads(num_threads_);

  ncnn::Mat decoder_out;
  decoder_ex.input("in0", decoder_input);
  decoder_ex.extract("out0", decoder_out);
  decoder_out = decoder_out.reshape(decoder_out.w);

  return decoder_out;
}

ncnn::Mat ConvEmformerModel::RunJoiner(ncnn::Mat &encoder_out,
                                       ncnn::Mat &decoder_out) {
  auto joiner_ex = joiner_.create_extractor();
  joiner_ex.set_num_threads(num_threads_);
  joiner_ex.input("in0", encoder_out);
  joiner_ex.input("in1", decoder_out);

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

void ConvEmformerModel::InitStateNames() {
  in_state_names_.clear();
  in_state_names_.reserve(num_layers_ * 4);

  out_state_names_.clear();
  out_state_names_.reserve(num_layers_ * 4);

  std::string in = "in";
  std::string out = "out";
  for (int32_t i = 0; i != num_layers_; ++i) {
    int32_t in_offset = 1 + i * 4;

    std::string name = in + std::to_string(in_offset);
    in_state_names_.push_back(std::move(name));

    name = in + std::to_string(in_offset + 1);
    in_state_names_.push_back(std::move(name));

    name = in + std::to_string(in_offset + 2);
    in_state_names_.push_back(std::move(name));

    name = in + std::to_string(in_offset + 3);
    in_state_names_.push_back(std::move(name));

    int32_t out_offset = 1 + i * 4;

    name = out + std::to_string(out_offset);
    out_state_names_.push_back(std::move(name));

    name = out + std::to_string(out_offset + 1);
    out_state_names_.push_back(std::move(name));

    name = out + std::to_string(out_offset + 2);
    out_state_names_.push_back(std::move(name));

    name = out + std::to_string(out_offset + 3);
    out_state_names_.push_back(std::move(name));
  }
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

}  // namespace sherpa_ncnn
