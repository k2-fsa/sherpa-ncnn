// sherpa-ncnn/csrc/zipformer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-ncnn/csrc/zipformer-model.h"

#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "net.h"       // NOLINT
#include "platform.h"  // NOLINT
#include "sherpa-ncnn/csrc/meta-data.h"

namespace sherpa_ncnn {

ZipformerModel::ZipformerModel(const ModelConfig &config) {
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
ZipformerModel::ZipformerModel(AAssetManager *mgr, const ModelConfig &config) {
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

std::pair<ncnn::Mat, std::vector<ncnn::Mat>> ZipformerModel::RunEncoder(
    ncnn::Mat &features, const std::vector<ncnn::Mat> &states) {
  ncnn::Extractor encoder_ex = encoder_.create_extractor();
  return RunEncoder(features, states, &encoder_ex);
}

std::pair<ncnn::Mat, std::vector<ncnn::Mat>> ZipformerModel::RunEncoder(
    ncnn::Mat &features, const std::vector<ncnn::Mat> &states,
    ncnn::Extractor *encoder_ex) {
  std::vector<ncnn::Mat> _states;

  const ncnn::Mat *p;
  if (states.empty()) {
    _states = GetEncoderInitStates();
    p = _states.data();
  } else {
    p = states.data();
  }

  // Note: We ignore error check there
  encoder_ex->input(encoder_input_indexes_[0], features);
  for (int32_t i = 1; i != encoder_input_indexes_.size(); ++i) {
    encoder_ex->input(encoder_input_indexes_[i], p[i - 1]);
  }

  ncnn::Mat encoder_out;
  encoder_ex->extract(encoder_output_indexes_[0], encoder_out);

  std::vector<ncnn::Mat> next_states(num_encoder_layers_.size() * 7);
  for (int32_t i = 1; i != encoder_output_indexes_.size(); ++i) {
    encoder_ex->extract(encoder_output_indexes_[i], next_states[i - 1]);
  }

  // reshape cached_avg to 1-D tensors; remove the w dim, which is 1
  for (size_t i = 0; i != num_encoder_layers_.size(); ++i) {
    next_states[i] = next_states[i].reshape(next_states[i].h);
  }

  // reshape cached_len to 2-D tensors, remove the h dim, which is 1
  for (size_t i = num_encoder_layers_.size();
       i != num_encoder_layers_.size() * 2; ++i) {
    next_states[i] = next_states[i].reshape(next_states[i].w, next_states[i].c);
  }

  return {encoder_out, next_states};
}

ncnn::Mat ZipformerModel::RunDecoder(ncnn::Mat &decoder_input) {
  ncnn::Extractor decoder_ex = decoder_.create_extractor();
  return RunDecoder(decoder_input, &decoder_ex);
}

ncnn::Mat ZipformerModel::RunDecoder(ncnn::Mat &decoder_input,
                                     ncnn::Extractor *decoder_ex) {
  ncnn::Mat decoder_out;
  decoder_ex->input(decoder_input_indexes_[0], decoder_input);
  decoder_ex->extract(decoder_output_indexes_[0], decoder_out);
  decoder_out = decoder_out.reshape(decoder_out.w);

  return decoder_out;
}

ncnn::Mat ZipformerModel::RunJoiner(ncnn::Mat &encoder_out,
                                    ncnn::Mat &decoder_out) {
  auto joiner_ex = joiner_.create_extractor();
  return RunJoiner(encoder_out, decoder_out, &joiner_ex);
}

ncnn::Mat ZipformerModel::RunJoiner(ncnn::Mat &encoder_out,
                                    ncnn::Mat &decoder_out,
                                    ncnn::Extractor *joiner_ex) {
  joiner_ex->input(joiner_input_indexes_[0], encoder_out);
  joiner_ex->input(joiner_input_indexes_[1], decoder_out);

  ncnn::Mat joiner_out;
  joiner_ex->extract(joiner_output_indexes_[0], joiner_out);
  return joiner_out;
}

void ZipformerModel::InitEncoderPostProcessing() {
  // Now load parameters for member variables
  for (const auto *layer : encoder_.layers()) {
    if (layer->type == "SherpaMetaData" && layer->name == "sherpa_meta_data1") {
      // Note: We don't use dynamic_cast<> here since it will throw
      // the following error
      //  error: ‘dynamic_cast’ not permitted with -fno-rtti
      const auto *meta_data = reinterpret_cast<const MetaData *>(layer);

      decode_chunk_length_ = meta_data->arg1;
      num_left_chunks_ = meta_data->arg2;
      pad_length_ = meta_data->arg3;

      num_encoder_layers_ = std::vector<int32_t>(
          static_cast<const int32_t *>(meta_data->arg16),
          static_cast<const int32_t *>(meta_data->arg16) + meta_data->arg16.w);

      encoder_dims_ = std::vector<int32_t>(
          static_cast<const int32_t *>(meta_data->arg17),
          static_cast<const int32_t *>(meta_data->arg17) + meta_data->arg17.w);

      attention_dims_ = std::vector<int32_t>(
          static_cast<const int32_t *>(meta_data->arg18),
          static_cast<const int32_t *>(meta_data->arg18) + meta_data->arg18.w);

      zipformer_downsampling_factors_ = std::vector<int32_t>(
          static_cast<const int32_t *>(meta_data->arg19),
          static_cast<const int32_t *>(meta_data->arg19) + meta_data->arg19.w);

      cnn_module_kernels_ = std::vector<int32_t>(
          static_cast<const int32_t *>(meta_data->arg20),
          static_cast<const int32_t *>(meta_data->arg20) + meta_data->arg20.w);

      break;
    }
  }
}

void ZipformerModel::InitEncoder(const std::string &encoder_param,
                                 const std::string &encoder_bin) {
  RegisterCustomLayers(encoder_);
  InitNet(encoder_, encoder_param, encoder_bin);
  InitEncoderPostProcessing();
}

void ZipformerModel::InitDecoder(const std::string &decoder_param,
                                 const std::string &decoder_bin) {
  InitNet(decoder_, decoder_param, decoder_bin);
}

void ZipformerModel::InitJoiner(const std::string &joiner_param,
                                const std::string &joiner_bin) {
  InitNet(joiner_, joiner_param, joiner_bin);
}

#if __ANDROID_API__ >= 9
void ZipformerModel::InitEncoder(AAssetManager *mgr,
                                 const std::string &encoder_param,
                                 const std::string &encoder_bin) {
  RegisterCustomLayers(encoder_);
  InitNet(mgr, encoder_, encoder_param, encoder_bin);
  InitEncoderPostProcessing();
}

void ZipformerModel::InitDecoder(AAssetManager *mgr,
                                 const std::string &decoder_param,
                                 const std::string &decoder_bin) {
  InitNet(mgr, decoder_, decoder_param, decoder_bin);
}

void ZipformerModel::InitJoiner(AAssetManager *mgr,
                                const std::string &joiner_param,
                                const std::string &joiner_bin) {
  InitNet(mgr, joiner_, joiner_param, joiner_bin);
}
#endif

// see
// https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/zipformer.py#L673
std::vector<ncnn::Mat> ZipformerModel::GetEncoderInitStates() const {
  // each layer has 7 states:
  // cached_len, (num_layers,)
  // cached_avg, (num_layers, encoder_dim)
  // cached_key, (num_layers, left_context_length, attention_dim)
  // cached_val, (num_layers, left_context_length, attention_dim / 2)
  // cached_val2, (num_layers, left_context_length, attention_dim / 2)
  // cached_conv1, (num_layers, encoder_dim, cnn_module_kernel_ - 1)
  // cached_conv2, (num_layers, encoder_dim, cnn_module_kernel_ - 1)

  std::vector<ncnn::Mat> cached_len_vec;
  std::vector<ncnn::Mat> cached_avg_vec;
  std::vector<ncnn::Mat> cached_key_vec;
  std::vector<ncnn::Mat> cached_val_vec;
  std::vector<ncnn::Mat> cached_val2_vec;
  std::vector<ncnn::Mat> cached_conv1_vec;
  std::vector<ncnn::Mat> cached_conv2_vec;

  cached_len_vec.reserve(num_encoder_layers_.size());
  cached_avg_vec.reserve(num_encoder_layers_.size());
  cached_key_vec.reserve(num_encoder_layers_.size());
  cached_val_vec.reserve(num_encoder_layers_.size());
  cached_val2_vec.reserve(num_encoder_layers_.size());
  cached_conv1_vec.reserve(num_encoder_layers_.size());
  cached_conv2_vec.reserve(num_encoder_layers_.size());

  int32_t left_context_length = decode_chunk_length_ / 2 * num_left_chunks_;
  for (size_t i = 0; i != num_encoder_layers_.size(); ++i) {
    int32_t num_layers = num_encoder_layers_[i];
    int32_t ds = zipformer_downsampling_factors_[i];
    int32_t attention_dim = attention_dims_[i];
    int32_t left_context_len = left_context_length / ds;
    int32_t encoder_dim = encoder_dims_[i];
    int32_t cnn_module_kernel = cnn_module_kernels_[i];

    auto cached_len = ncnn::Mat(num_layers);
    auto cached_avg = ncnn::Mat(encoder_dim, num_layers);
    auto cached_key = ncnn::Mat(attention_dim, left_context_len, num_layers);
    auto cached_val =
        ncnn::Mat(attention_dim / 2, left_context_len, num_layers);
    auto cached_val2 =
        ncnn::Mat(attention_dim / 2, left_context_len, num_layers);
    auto cached_conv1 =
        ncnn::Mat(cnn_module_kernel - 1, encoder_dim, num_layers);
    auto cached_conv2 =
        ncnn::Mat(cnn_module_kernel - 1, encoder_dim, num_layers);

    cached_len.fill(0);
    cached_avg.fill(0);
    cached_key.fill(0);
    cached_val.fill(0);
    cached_val2.fill(0);
    cached_conv1.fill(0);
    cached_conv2.fill(0);

    cached_len_vec.push_back(cached_len);
    cached_avg_vec.push_back(cached_avg);
    cached_key_vec.push_back(cached_key);
    cached_val_vec.push_back(cached_val);
    cached_val2_vec.push_back(cached_val2);
    cached_conv1_vec.push_back(cached_conv1);
    cached_conv2_vec.push_back(cached_conv2);
  }

  std::vector<ncnn::Mat> states;

  states.reserve(num_encoder_layers_.size() * 7);
  states.insert(states.end(), cached_len_vec.begin(), cached_len_vec.end());
  states.insert(states.end(), cached_avg_vec.begin(), cached_avg_vec.end());
  states.insert(states.end(), cached_key_vec.begin(), cached_key_vec.end());
  states.insert(states.end(), cached_val_vec.begin(), cached_val_vec.end());
  states.insert(states.end(), cached_val2_vec.begin(), cached_val2_vec.end());
  states.insert(states.end(), cached_conv1_vec.begin(), cached_conv1_vec.end());
  states.insert(states.end(), cached_conv2_vec.begin(), cached_conv2_vec.end());

  return states;
}

void ZipformerModel::InitEncoderInputOutputIndexes() {
  // input indexes map
  // [0] -> in0, features,
  // [1] -> in1, layer0, cached_len
  // [2] -> in2, layer1, cached_len
  // [3] -> in3, layer2, cached_len
  // ... ...
  encoder_input_indexes_.resize(1 + num_encoder_layers_.size() * 7);

  // output indexes map
  // [0] -> out0, encoder_out
  //
  // [1] -> out1, layer0, cached_len
  // [2] -> out2, layer1, cached_len
  // [3] -> out3, layer2, cached_len
  // ... ...
  encoder_output_indexes_.resize(1 + num_encoder_layers_.size() * 7);
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

void ZipformerModel::InitDecoderInputOutputIndexes() {
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

void ZipformerModel::InitJoinerInputOutputIndexes() {
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
