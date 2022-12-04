/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sherpa-ncnn/csrc/lstm-model.h"

#include <utility>
#include <vector>

namespace sherpa_ncnn {

LstmModel::LstmModel(const ModelConfig &config)
    : num_threads_(config.num_threads) {
  InitEncoder(config.encoder_param, config.encoder_bin);
  InitDecoder(config.decoder_param, config.decoder_bin);
  InitJoiner(config.joiner_param, config.joiner_bin);
}

std::pair<ncnn::Mat, std::vector<ncnn::Mat>> LstmModel::RunEncoder(
    ncnn::Mat &features, const std::vector<ncnn::Mat> &states) {
  ncnn::Mat hx;
  ncnn::Mat cx;

  if (states.empty()) {
    auto s = GetEncoderInitStates();
    hx = s[0];
    cx = s[1];
  } else {
    hx = states[0];
    cx = states[1];
  }

  ncnn::Mat feature_lengths(1);
  feature_lengths[0] = features.h;

  ncnn::Extractor encoder_ex = encoder_.create_extractor();
  encoder_ex.set_num_threads(num_threads_);

  encoder_ex.input("in0", features);
  encoder_ex.input("in1", feature_lengths);
  encoder_ex.input("in2", hx);
  encoder_ex.input("in3", cx);

  ncnn::Mat encoder_out;
  encoder_ex.extract("out0", encoder_out);

  encoder_ex.extract("out2", hx);
  encoder_ex.extract("out3", cx);

  std::vector<ncnn::Mat> next_states = {hx, cx};

  return {encoder_out, next_states};
}

ncnn::Mat LstmModel::RunDecoder(ncnn::Mat &decoder_input) {
  ncnn::Extractor decoder_ex = decoder_.create_extractor();
  decoder_ex.set_num_threads(num_threads_);

  ncnn::Mat decoder_out;
  decoder_ex.input("in0", decoder_input);
  decoder_ex.extract("out0", decoder_out);
  decoder_out = decoder_out.reshape(decoder_out.w);

  return decoder_out;
}

ncnn::Mat LstmModel::RunJoiner(ncnn::Mat &encoder_out, ncnn::Mat &decoder_out) {
  auto joiner_ex = joiner_.create_extractor();
  joiner_ex.set_num_threads(num_threads_);
  joiner_ex.input("in0", encoder_out);
  joiner_ex.input("in1", decoder_out);

  ncnn::Mat joiner_out;
  joiner_ex.extract("out0", joiner_out);
  return joiner_out;
}

void LstmModel::InitEncoder(const std::string &encoder_param,
                            const std::string &encoder_bin) {
  encoder_.opt.use_packing_layout = false;
  encoder_.opt.use_fp16_storage = false;
  InitNet(encoder_, encoder_param, encoder_bin);
}

void LstmModel::InitDecoder(const std::string &decoder_param,
                            const std::string &decoder_bin) {
  InitNet(decoder_, decoder_param, decoder_bin);
}

void LstmModel::InitJoiner(const std::string &joiner_param,
                           const std::string &joiner_bin) {
  InitNet(joiner_, joiner_param, joiner_bin);
}

std::vector<ncnn::Mat> LstmModel::GetEncoderInitStates() const {
  int32_t num_encoder_layers = 12;
  int32_t d_model = 512;
  int32_t rnn_hidden_size = 1024;

  auto hx = ncnn::Mat(d_model, num_encoder_layers);
  auto cx = ncnn::Mat(rnn_hidden_size, num_encoder_layers);

  hx.fill(0);
  cx.fill(0);

  return {hx, cx};
}

}  // namespace sherpa_ncnn
