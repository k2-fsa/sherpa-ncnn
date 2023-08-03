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

#include "platform.h"  // NOLINT
#include "sherpa-ncnn/csrc/meta-data.h"

namespace sherpa_ncnn {

LstmModel::LstmModel(const ModelConfig &config) {
  encoder_.opt = config.encoder_opt;
  decoder_.opt = config.decoder_opt;
  joiner_.opt = config.joiner_opt;

  bool has_gpu = false;
#if NCNN_VULKAN
  has_gpu = ncnn::get_gpu_count() > 0;
#endif

  if (has_gpu && config.use_vulkan_compute) {
    encoder_.opt.use_vulkan_compute = true;
    decoder_.opt.use_vulkan_compute = true;
    joiner_.opt.use_vulkan_compute = true;
    NCNN_LOGE("Use GPU");
  } else {
    // NCNN_LOGE("Don't Use GPU. has_gpu: %d, config.use_vulkan_compute: %d",
    //           static_cast<int32_t>(has_gpu),
    //           static_cast<int32_t>(config.use_vulkan_compute));
  }

  InitEncoder(config.encoder_param, config.encoder_bin);
  InitDecoder(config.decoder_param, config.decoder_bin);
  InitJoiner(config.joiner_param, config.joiner_bin);

  InitEncoderInputOutputIndexes();
  InitDecoderInputOutputIndexes();
  InitJoinerInputOutputIndexes();
}

#if __ANDROID_API__ >= 9
LstmModel::LstmModel(AAssetManager *mgr, const ModelConfig &config) {
  encoder_.opt = config.encoder_opt;
  decoder_.opt = config.decoder_opt;
  joiner_.opt = config.joiner_opt;

  bool has_gpu = false;
#if NCNN_VULKAN
  has_gpu = ncnn::get_gpu_count() > 0;
#endif

  if (has_gpu && config.use_vulkan_compute) {
    encoder_.opt.use_vulkan_compute = true;
    decoder_.opt.use_vulkan_compute = true;
    joiner_.opt.use_vulkan_compute = true;
    NCNN_LOGE("Use GPU");
  } else {
    // NCNN_LOGE("Don't Use GPU. has_gpu: %d, config.use_vulkan_compute: %d",
    //           static_cast<int32_t>(has_gpu),
    //           static_cast<int32_t>(config.use_vulkan_compute));
  }

  InitEncoder(mgr, config.encoder_param, config.encoder_bin);
  InitDecoder(mgr, config.decoder_param, config.decoder_bin);
  InitJoiner(mgr, config.joiner_param, config.joiner_bin);

  InitEncoderInputOutputIndexes();
  InitDecoderInputOutputIndexes();
  InitJoinerInputOutputIndexes();
}
#endif

std::pair<ncnn::Mat, std::vector<ncnn::Mat>> LstmModel::RunEncoder(
    ncnn::Mat &features, const std::vector<ncnn::Mat> &states,
    ncnn::Extractor *encoder_ex) {
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

  ncnn::Mat feature_length(1);
  feature_length[0] = features.h;

  encoder_ex->input(encoder_input_indexes_[0], features);
  encoder_ex->input(encoder_input_indexes_[1], feature_length);
  encoder_ex->input(encoder_input_indexes_[2], hx);
  encoder_ex->input(encoder_input_indexes_[3], cx);

  ncnn::Mat encoder_out;
  encoder_ex->extract(encoder_output_indexes_[0], encoder_out);

  encoder_ex->extract(encoder_output_indexes_[1], hx);
  encoder_ex->extract(encoder_output_indexes_[2], cx);

  std::vector<ncnn::Mat> next_states = {hx, cx};

  return {encoder_out, next_states};
}

std::pair<ncnn::Mat, std::vector<ncnn::Mat>> LstmModel::RunEncoder(
    ncnn::Mat &features, const std::vector<ncnn::Mat> &states) {
  ncnn::Extractor encoder_ex = encoder_.create_extractor();
  return RunEncoder(features, states, &encoder_ex);
}

ncnn::Mat LstmModel::RunDecoder(ncnn::Mat &decoder_input) {
  ncnn::Extractor decoder_ex = decoder_.create_extractor();
  return RunDecoder(decoder_input, &decoder_ex);
}

ncnn::Mat LstmModel::RunDecoder(ncnn::Mat &decoder_input,
                                ncnn::Extractor *decoder_ex) {
  ncnn::Mat decoder_out;
  decoder_ex->input(decoder_input_indexes_[0], decoder_input);
  decoder_ex->extract(decoder_output_indexes_[0], decoder_out);
  decoder_out = decoder_out.reshape(decoder_out.w);

  return decoder_out;
}

ncnn::Mat LstmModel::RunJoiner(ncnn::Mat &encoder_out, ncnn::Mat &decoder_out) {
  auto joiner_ex = joiner_.create_extractor();
  return RunJoiner(encoder_out, decoder_out, &joiner_ex);
}

ncnn::Mat LstmModel::RunJoiner(ncnn::Mat &encoder_out, ncnn::Mat &decoder_out,
                               ncnn::Extractor *joiner_ex) {
  joiner_ex->input(joiner_input_indexes_[0], encoder_out);
  joiner_ex->input(joiner_input_indexes_[1], decoder_out);

  ncnn::Mat joiner_out;
  joiner_ex->extract(joiner_output_indexes_[0], joiner_out);
  return joiner_out;
}

void LstmModel::InitEncoder(const std::string &encoder_param,
                            const std::string &encoder_bin) {
  RegisterCustomLayers(encoder_);
  InitNet(encoder_, encoder_param, encoder_bin);

  InitEncoderPostProcessing();
}

void LstmModel::InitDecoder(const std::string &decoder_param,
                            const std::string &decoder_bin) {
  InitNet(decoder_, decoder_param, decoder_bin);
}

void LstmModel::InitJoiner(const std::string &joiner_param,
                           const std::string &joiner_bin) {
  InitNet(joiner_, joiner_param, joiner_bin);
}

#if __ANDROID_API__ >= 9
void LstmModel::InitEncoder(AAssetManager *mgr,
                            const std::string &encoder_param,
                            const std::string &encoder_bin) {
  RegisterCustomLayers(encoder_);
  InitNet(mgr, encoder_, encoder_param, encoder_bin);

  InitEncoderPostProcessing();
}

void LstmModel::InitDecoder(AAssetManager *mgr,
                            const std::string &decoder_param,
                            const std::string &decoder_bin) {
  InitNet(mgr, decoder_, decoder_param, decoder_bin);
}

void LstmModel::InitJoiner(AAssetManager *mgr, const std::string &joiner_param,
                           const std::string &joiner_bin) {
  InitNet(mgr, joiner_, joiner_param, joiner_bin);
}
#endif

void LstmModel::InitEncoderPostProcessing() {
  // Now load parameters for member variables
  for (const auto *layer : encoder_.layers()) {
    if (layer->type == "SherpaMetaData" && layer->name == "sherpa_meta_data1") {
      // Note: We don't use dynamic_cast<> here since it will throw
      // the following error
      //  error: ‘dynamic_cast’ not permitted with -fno-rtti
      const auto *meta_data = reinterpret_cast<const MetaData *>(layer);

      num_encoder_layers_ = meta_data->arg1;
      encoder_dim_ = meta_data->arg2;
      rnn_hidden_size_ = meta_data->arg3;

      break;
    }
  }
}

std::vector<ncnn::Mat> LstmModel::GetEncoderInitStates() const {
  auto hx = ncnn::Mat(encoder_dim_, num_encoder_layers_);
  auto cx = ncnn::Mat(rnn_hidden_size_, num_encoder_layers_);

  hx.fill(0);
  cx.fill(0);

  return {hx, cx};
}

void LstmModel::InitEncoderInputOutputIndexes() {
  // input indexes map
  // [0] -> in0, features,
  // [1] -> in1, features_length
  // [2] -> in2, hx
  // [3] -> in3, cx
  encoder_input_indexes_.resize(4);

  // output indexes map
  // [0] -> out0, encoder_out
  // [1] -> out2, hx
  // [2] -> out3, cx
  encoder_output_indexes_.resize(3);
  const auto &blobs = encoder_.blobs();
  for (int32_t i = 0; i != blobs.size(); ++i) {
    const auto &b = blobs[i];
    if (b.name == "in0") encoder_input_indexes_[0] = i;
    if (b.name == "in1") encoder_input_indexes_[1] = i;
    if (b.name == "in2") encoder_input_indexes_[2] = i;
    if (b.name == "in3") encoder_input_indexes_[3] = i;
    if (b.name == "out0") encoder_output_indexes_[0] = i;
    if (b.name == "out2") encoder_output_indexes_[1] = i;
    if (b.name == "out3") encoder_output_indexes_[2] = i;
  }
}

void LstmModel::InitDecoderInputOutputIndexes() {
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

void LstmModel::InitJoinerInputOutputIndexes() {
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
