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

/* This file is modified from
 https://github.com/Tencent/ncnn/blob/master/tools/quantize/ncnn2table.cpp
*/

#include <float.h>
#include <stdio.h>  // for FLT_MAX

#include <fstream>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "layer/convolution.h"
#include "layer/convolutiondepthwise.h"
#include "layer/innerproduct.h"
#include "mat.h"
#include "net.h"
#include "sherpa-ncnn/csrc/features.h"
#include "sherpa-ncnn/csrc/model.h"
#include "sherpa-ncnn/csrc/recognizer.h"
#include "sherpa-ncnn/csrc/wave-reader.h"

static float compute_kl_divergence(const std::vector<float> &a,
                                   const std::vector<float> &b) {
  const size_t length = a.size();

  float result = 0;
  for (size_t i = 0; i < length; i++) {
    result += a[i] * log(a[i] / b[i]);
  }

  return result;
}

class QuantBlobStat {
 public:
  QuantBlobStat() {
    threshold = 0.f;
    absmax = 0.f;
    total = 0;
  }

 public:
  float threshold;
  float absmax;

  // ACIQ
  int total;

  // KL
  std::vector<uint64_t> histogram;
  std::vector<float> histogram_normed;
};

float compute_kl_threshold(QuantBlobStat &stat, int num_histogram_bins = 2048) {
  // normalize histogram bin
  {
    uint64_t sum = 0;
    for (int j = 0; j < num_histogram_bins; j++) {
      sum += stat.histogram[j];
    }

    for (int j = 0; j < num_histogram_bins; j++) {
      stat.histogram_normed[j] = (float)(stat.histogram[j] / (double)sum);
    }
  }

  const int target_bin = 128;

  int target_threshold = target_bin;
  float min_kl_divergence = FLT_MAX;

  for (int threshold = target_bin; threshold < num_histogram_bins;
       threshold++) {
    const float kl_eps = 0.0001f;

    std::vector<float> clip_distribution(threshold, kl_eps);
    {
      for (int j = 0; j < threshold; j++) {
        clip_distribution[j] += stat.histogram_normed[j];
      }
      for (int j = threshold; j < num_histogram_bins; j++) {
        clip_distribution[threshold - 1] += stat.histogram_normed[j];
      }
    }

    const float num_per_bin = (float)threshold / target_bin;

    std::vector<float> quantize_distribution(target_bin, 0.f);
    {
      {
        const float end = num_per_bin;

        const int right_lower = (int)floor(end);
        const float right_scale = end - right_lower;

        if (right_scale > 0) {
          quantize_distribution[0] +=
              right_scale * stat.histogram_normed[right_lower];
        }

        for (int k = 0; k < right_lower; k++) {
          quantize_distribution[0] += stat.histogram_normed[k];
        }

        quantize_distribution[0] /= right_lower + right_scale;
      }
      for (int j = 1; j < target_bin - 1; j++) {
        const float start = j * num_per_bin;
        const float end = (j + 1) * num_per_bin;

        const int left_upper = (int)ceil(start);
        const float left_scale = left_upper - start;

        const int right_lower = (int)floor(end);
        const float right_scale = end - right_lower;

        if (left_scale > 0) {
          quantize_distribution[j] +=
              left_scale * stat.histogram_normed[left_upper - 1];
        }

        if (right_scale > 0) {
          quantize_distribution[j] +=
              right_scale * stat.histogram_normed[right_lower];
        }

        for (int k = left_upper; k < right_lower; k++) {
          quantize_distribution[j] += stat.histogram_normed[k];
        }

        quantize_distribution[j] /=
            right_lower - left_upper + left_scale + right_scale;
      }
      {
        const float start = threshold - num_per_bin;

        const int left_upper = (int)ceil(start);
        const float left_scale = left_upper - start;

        if (left_scale > 0) {
          quantize_distribution[target_bin - 1] +=
              left_scale * stat.histogram_normed[left_upper - 1];
        }

        for (int k = left_upper; k < threshold; k++) {
          quantize_distribution[target_bin - 1] += stat.histogram_normed[k];
        }

        quantize_distribution[target_bin - 1] /=
            threshold - left_upper + left_scale;
      }
    }

    std::vector<float> expand_distribution(threshold, kl_eps);
    {
      {
        const float end = num_per_bin;

        const int right_lower = (int)floor(end);
        const float right_scale = end - right_lower;

        if (right_scale > 0) {
          expand_distribution[right_lower] +=
              right_scale * quantize_distribution[0];
        }

        for (int k = 0; k < right_lower; k++) {
          expand_distribution[k] += quantize_distribution[0];
        }
      }
      for (int j = 1; j < target_bin - 1; j++) {
        const float start = j * num_per_bin;
        const float end = (j + 1) * num_per_bin;

        const int left_upper = (int)ceil(start);
        const float left_scale = left_upper - start;

        const int right_lower = (int)floor(end);
        const float right_scale = end - right_lower;

        if (left_scale > 0) {
          expand_distribution[left_upper - 1] +=
              left_scale * quantize_distribution[j];
        }

        if (right_scale > 0) {
          expand_distribution[right_lower] +=
              right_scale * quantize_distribution[j];
        }

        for (int k = left_upper; k < right_lower; k++) {
          expand_distribution[k] += quantize_distribution[j];
        }
      }
      {
        const float start = threshold - num_per_bin;

        const int left_upper = (int)ceil(start);
        const float left_scale = left_upper - start;

        if (left_scale > 0) {
          expand_distribution[left_upper - 1] +=
              left_scale * quantize_distribution[target_bin - 1];
        }

        for (int k = left_upper; k < threshold; k++) {
          expand_distribution[k] += quantize_distribution[target_bin - 1];
        }
      }
    }

    // kl
    const float kl_divergence =
        compute_kl_divergence(clip_distribution, expand_distribution);

    // the best num of bin
    if (kl_divergence < min_kl_divergence) {
      min_kl_divergence = kl_divergence;
      target_threshold = threshold;
    }
  }

  stat.threshold = (target_threshold + 0.5f) * stat.absmax / num_histogram_bins;
  float scale = 127 / stat.threshold;

  return scale;
}

class QuantNet : public ncnn::Net {
 public:
  QuantNet(sherpa_ncnn::Model *model);

  sherpa_ncnn::Model *model;
  std::vector<ncnn::Layer *> &encoder_layers;
  std::vector<ncnn::Layer *> &joiner_layers;

 public:
  int init();
  void print_quant_info() const;
  int save_table_encoder(const char *tablepath);
  int save_table_joiner(const char *tablepath);
  int quantize_KL(const std::vector<std::string> &wave_filenames);
  int quantize_ACIQ();
  int quantize_EQ();

 private:
  int init_encoder();
  int init_joiner();

  void quantize_encoder_weight();
  void quantize_joiner_weight();

 public:
  std::vector<int> encoder_conv_layers;
  std::vector<int> encoder_conv_bottom_blobs;

  std::vector<int> joiner_conv_layers;
  std::vector<int> joiner_conv_bottom_blobs;

  // result
  std::vector<QuantBlobStat> encoder_quant_blob_stats;
  std::vector<ncnn::Mat> encoder_weight_scales;
  std::vector<ncnn::Mat> encoder_bottom_blob_scales;

  std::vector<QuantBlobStat> joiner_quant_blob_stats;
  std::vector<ncnn::Mat> joiner_weight_scales;
  std::vector<ncnn::Mat> joiner_bottom_blob_scales;
};

QuantNet::QuantNet(sherpa_ncnn::Model *model)
    : model(model),
      encoder_layers(model->GetEncoder().mutable_layers()),
      joiner_layers(model->GetJoiner().mutable_layers()) {}

int QuantNet::init_encoder() {
  // find all encoder conv layers
  for (int i = 0; i < (int)encoder_layers.size(); i++) {
    const ncnn::Layer *layer = encoder_layers[i];
    if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise" ||
        layer->type == "InnerProduct") {
      encoder_conv_layers.push_back(i);
      encoder_conv_bottom_blobs.push_back(layer->bottoms[0]);
    }
  }

  fprintf(stderr, "num encoder conv layers: %d\n",
          static_cast<int32_t>(encoder_conv_layers.size()));

  const int encoder_conv_layer_count = (int)encoder_conv_layers.size();
  const int encoder_conv_bottom_blob_count =
      (int)encoder_conv_bottom_blobs.size();

  encoder_quant_blob_stats.resize(encoder_conv_bottom_blob_count);
  encoder_weight_scales.resize(encoder_conv_layer_count);
  encoder_bottom_blob_scales.resize(encoder_conv_bottom_blob_count);

  return 0;
}

int QuantNet::init_joiner() {
  // find all joiner conv layers
  for (int i = 0; i < (int)joiner_layers.size(); i++) {
    const ncnn::Layer *layer = joiner_layers[i];
    if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise" ||
        layer->type == "InnerProduct") {
      joiner_conv_layers.push_back(i);
      joiner_conv_bottom_blobs.push_back(layer->bottoms[0]);
    }
  }

  fprintf(stderr, "num joiner conv layers: %d\n",
          static_cast<int32_t>(joiner_conv_layers.size()));

  const int joiner_conv_layer_count = (int)joiner_conv_layers.size();
  const int joiner_conv_bottom_blob_count =
      (int)joiner_conv_bottom_blobs.size();

  joiner_quant_blob_stats.resize(joiner_conv_bottom_blob_count);
  joiner_weight_scales.resize(joiner_conv_layer_count);
  joiner_bottom_blob_scales.resize(joiner_conv_bottom_blob_count);

  return 0;
}

int QuantNet::init() {
  init_encoder();
  init_joiner();

  return 0;
}

void QuantNet::quantize_encoder_weight() {
  const int encoder_conv_layer_count = (int)encoder_conv_layers.size();

  for (int i = 0; i < encoder_conv_layer_count; i++) {
    const ncnn::Layer *layer = encoder_layers[encoder_conv_layers[i]];
    if (layer->type == "Convolution") {
      const ncnn::Convolution *convolution = (const ncnn::Convolution *)layer;
      const int num_output = convolution->num_output;
      const int kernel_w = convolution->kernel_w;
      const int kernel_h = convolution->kernel_h;
      const int dilation_w = convolution->dilation_w;
      const int dilation_h = convolution->dilation_h;
      const int stride_w = convolution->stride_w;
      const int stride_h = convolution->stride_h;

      const int weight_data_size_output =
          convolution->weight_data_size / num_output;

      // int8 winograd F43 needs weight data to use 6bit quantization
      // TODO proper condition for winograd 3x3 int8
      bool quant_6bit = false;
      if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 &&
          dilation_h == 1 && stride_w == 1 && stride_h == 1) {
        quant_6bit = true;
      }

      encoder_weight_scales[i].create(num_output);
      for (int n = 0; n < num_output; n++) {
        const ncnn::Mat weight_data_n = convolution->weight_data.range(
            weight_data_size_output * n, weight_data_size_output);

        float absmax = 0.f;
        for (int k = 0; k < weight_data_size_output; k++) {
          absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
        }

        if (quant_6bit) {
          encoder_weight_scales[i][n] = 31 / absmax;
        } else {
          encoder_weight_scales[i][n] = 127 / absmax;
        }
      }
    }  // if (layer->type == "Convolution")

    if (layer->type == "ConvolutionDepthWise") {
      const ncnn::ConvolutionDepthWise *convolutiondepthwise =
          (const ncnn::ConvolutionDepthWise *)layer;

      const int group = convolutiondepthwise->group;
      const int weight_data_size_output =
          convolutiondepthwise->weight_data_size / group;

      std::vector<float> scales;

      encoder_weight_scales[i].create(group);

      for (int n = 0; n < group; n++) {
        const ncnn::Mat weight_data_n = convolutiondepthwise->weight_data.range(
            weight_data_size_output * n, weight_data_size_output);

        float absmax = 0.f;
        for (int k = 0; k < weight_data_size_output; k++) {
          absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
        }

        encoder_weight_scales[i][n] = 127 / absmax;
      }
    }  // if (layer->type == "ConvolutionDepthWise")

    if (layer->type == "InnerProduct") {
      const ncnn::InnerProduct *innerproduct =
          (const ncnn::InnerProduct *)layer;

      const int num_output = innerproduct->num_output;
      const int weight_data_size_output =
          innerproduct->weight_data_size / num_output;

      encoder_weight_scales[i].create(num_output);

      for (int n = 0; n < num_output; n++) {
        const ncnn::Mat weight_data_n = innerproduct->weight_data.range(
            weight_data_size_output * n, weight_data_size_output);

        float absmax = 0.f;
        for (int k = 0; k < weight_data_size_output; k++) {
          absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
        }

        encoder_weight_scales[i][n] = 127 / absmax;
      }
    }  // if (layer->type == "InnerProduct")
  }    // for (int i = 0; i < encoder_conv_layer_count; i++)
}

void QuantNet::quantize_joiner_weight() {
  const int joiner_conv_layer_count = (int)joiner_conv_layers.size();

  for (int i = 0; i < joiner_conv_layer_count; i++) {
    const ncnn::Layer *layer = joiner_layers[joiner_conv_layers[i]];
    if (layer->type == "Convolution") {
      const ncnn::Convolution *convolution = (const ncnn::Convolution *)layer;
      const int num_output = convolution->num_output;
      const int kernel_w = convolution->kernel_w;
      const int kernel_h = convolution->kernel_h;
      const int dilation_w = convolution->dilation_w;
      const int dilation_h = convolution->dilation_h;
      const int stride_w = convolution->stride_w;
      const int stride_h = convolution->stride_h;

      const int weight_data_size_output =
          convolution->weight_data_size / num_output;

      // int8 winograd F43 needs weight data to use 6bit quantization
      // TODO proper condition for winograd 3x3 int8
      bool quant_6bit = false;
      if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 &&
          dilation_h == 1 && stride_w == 1 && stride_h == 1) {
        quant_6bit = true;
      }

      joiner_weight_scales[i].create(num_output);
      for (int n = 0; n < num_output; n++) {
        const ncnn::Mat weight_data_n = convolution->weight_data.range(
            weight_data_size_output * n, weight_data_size_output);

        float absmax = 0.f;
        for (int k = 0; k < weight_data_size_output; k++) {
          absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
        }

        if (quant_6bit) {
          joiner_weight_scales[i][n] = 31 / absmax;
        } else {
          joiner_weight_scales[i][n] = 127 / absmax;
        }
      }
    }  // if (layer->type == "Convolution")

    if (layer->type == "ConvolutionDepthWise") {
      const ncnn::ConvolutionDepthWise *convolutiondepthwise =
          (const ncnn::ConvolutionDepthWise *)layer;

      const int group = convolutiondepthwise->group;
      const int weight_data_size_output =
          convolutiondepthwise->weight_data_size / group;

      std::vector<float> scales;

      joiner_weight_scales[i].create(group);

      for (int n = 0; n < group; n++) {
        const ncnn::Mat weight_data_n = convolutiondepthwise->weight_data.range(
            weight_data_size_output * n, weight_data_size_output);

        float absmax = 0.f;
        for (int k = 0; k < weight_data_size_output; k++) {
          absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
        }

        joiner_weight_scales[i][n] = 127 / absmax;
      }
    }  // if (layer->type == "ConvolutionDepthWise")

    if (layer->type == "InnerProduct") {
      const ncnn::InnerProduct *innerproduct =
          (const ncnn::InnerProduct *)layer;

      const int num_output = innerproduct->num_output;
      const int weight_data_size_output =
          innerproduct->weight_data_size / num_output;

      joiner_weight_scales[i].create(num_output);

      for (int n = 0; n < num_output; n++) {
        const ncnn::Mat weight_data_n = innerproduct->weight_data.range(
            weight_data_size_output * n, weight_data_size_output);

        float absmax = 0.f;
        for (int k = 0; k < weight_data_size_output; k++) {
          absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
        }

        joiner_weight_scales[i][n] = 127 / absmax;
      }
    }  // if (layer->type == "InnerProduct")
  }    // for (int i = 0; i < joiner_conv_layer_count; i++)
}

int QuantNet::quantize_KL(const std::vector<std::string> &wave_filenames) {
  const int encoder_conv_bottom_blob_count =
      (int)encoder_conv_bottom_blobs.size();

  const int joiner_conv_bottom_blob_count =
      (int)joiner_conv_bottom_blobs.size();

  fprintf(stderr, "num files: %d\n", (int)wave_filenames.size());

  const int num_histogram_bins = 2048;
  std::vector<ncnn::UnlockedPoolAllocator> blob_allocators(1);
  std::vector<ncnn::UnlockedPoolAllocator> workspace_allocators(1);

  // initialize conv weight scales
  quantize_encoder_weight();
  quantize_joiner_weight();

  float expected_sampling_rate = 16000;

  knf::FbankOptions fbank_opts;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.snip_edges = false;
  fbank_opts.frame_opts.samp_freq = expected_sampling_rate;
  fbank_opts.mel_opts.num_bins = 80;

  // Please see
  // https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/fbank.py#L27
  // and
  // https://github.com/k2-fsa/sherpa-onnx/issues/514
  fbank_opts.mel_opts.high_freq = -400;

  int32_t segment = model->Segment();
  int32_t offset = model->Offset();

  // count the absmax
  for (const auto &filename : wave_filenames) {
    bool is_ok = false;
    std::vector<float> samples =
        sherpa_ncnn::ReadWave(filename, expected_sampling_rate, &is_ok);
    if (!is_ok) {
      fprintf(stderr, "Failed to read %s\n", filename.c_str());
      continue;
    }
    fprintf(stderr, "Processing %s\n", filename.c_str());

    sherpa_ncnn::FeatureExtractorConfig config;
    config.sampling_rate = 16000;
    config.feature_dim = 80;
    sherpa_ncnn::FeatureExtractor feature_extractor(config);
    feature_extractor.AcceptWaveform(expected_sampling_rate, samples.data(),
                                     samples.size());
    feature_extractor.InputFinished();

    int32_t context_size = model->ContextSize();
    int32_t blank_id = model->BlankId();

    std::vector<int32_t> hyp(context_size, blank_id);

    ncnn::Mat decoder_input(context_size);
    for (int32_t i = 0; i != context_size; ++i) {
      static_cast<int32_t *>(decoder_input)[i] = blank_id;
    }

    ncnn::Mat decoder_out = model->RunDecoder(decoder_input);

    std::vector<ncnn::Mat> states;
    ncnn::Mat encoder_out;

    int32_t num_processed = 0;
    while (feature_extractor.NumFramesReady() - num_processed >= segment) {
      ncnn::Extractor encoder_ex = model->GetEncoder().create_extractor();
      encoder_ex.set_light_mode(false);
      encoder_ex.set_blob_allocator(&blob_allocators[0]);
      encoder_ex.set_workspace_allocator(&workspace_allocators[0]);

      ncnn::Extractor joiner_ex = model->GetJoiner().create_extractor();
      joiner_ex.set_light_mode(false);
      joiner_ex.set_blob_allocator(&blob_allocators[0]);
      joiner_ex.set_workspace_allocator(&workspace_allocators[0]);

      ncnn::Mat features = feature_extractor.GetFrames(num_processed, segment);
      num_processed += offset;
      std::tie(encoder_out, states) =
          model->RunEncoder(features, states, &encoder_ex);

      for (int j = 0; j < encoder_conv_bottom_blob_count; j++) {
        ncnn::Mat out;
        encoder_ex.extract(encoder_conv_bottom_blobs[j], out);

        {
          // count absmax
          float absmax = 0.f;

          const int outc = out.c;
          const int outsize = out.w * out.h;
          for (int p = 0; p < outc; p++) {
            const float *ptr = out.channel(p);
            for (int k = 0; k < outsize; k++) {
              absmax = std::max(absmax, (float)fabs(ptr[k]));
            }
          }

          QuantBlobStat &stat = encoder_quant_blob_stats[j];
          stat.absmax = std::max(stat.absmax, absmax);
        }
      }  // for (int j = 0; j < encoder_conv_bottom_blob_count; j++)

      // now for joiner
      for (int32_t t = 0; t != encoder_out.h; ++t) {
        ncnn::Mat encoder_out_t(encoder_out.w, encoder_out.row(t));
        ncnn::Mat joiner_out =
            model->RunJoiner(encoder_out_t, decoder_out, &joiner_ex);

        for (int j = 0; j < joiner_conv_bottom_blob_count; j++) {
          ncnn::Mat out;
          joiner_ex.extract(joiner_conv_bottom_blobs[j], out);

          {
            // count absmax
            float absmax = 0.f;

            const int outc = out.c;
            const int outsize = out.w * out.h;
            for (int p = 0; p < outc; p++) {
              const float *ptr = out.channel(p);
              for (int k = 0; k < outsize; k++) {
                absmax = std::max(absmax, (float)fabs(ptr[k]));
              }
            }

            QuantBlobStat &stat = joiner_quant_blob_stats[j];
            stat.absmax = std::max(stat.absmax, absmax);
          }
        }  // for (int j = 0; j < joiner_conv_bottom_blob_count; j++)

        auto y = static_cast<int32_t>(std::distance(
            static_cast<const float *>(joiner_out),
            std::max_element(
                static_cast<const float *>(joiner_out),
                static_cast<const float *>(joiner_out) + joiner_out.w)));

        if (y != blank_id) {
          static_cast<int32_t *>(decoder_input)[0] = hyp.back();
          static_cast<int32_t *>(decoder_input)[1] = y;
          hyp.push_back(y);

          decoder_out = model->RunDecoder(decoder_input);
        }
      }  // for (int32_t t = 0; t != encoder_out.h; ++t)

    }  // while (feature_extractor.NumFramesReady() - num_processed >=
       // segment)
  }    // for (const auto &filename : wave_filenames)

  // initialize histogram
  for (int i = 0; i < encoder_conv_bottom_blob_count; i++) {
    QuantBlobStat &stat = encoder_quant_blob_stats[i];

    stat.histogram.resize(num_histogram_bins, 0);
    stat.histogram_normed.resize(num_histogram_bins, 0);
  }

  for (int i = 0; i < joiner_conv_bottom_blob_count; i++) {
    QuantBlobStat &stat = joiner_quant_blob_stats[i];

    stat.histogram.resize(num_histogram_bins, 0);
    stat.histogram_normed.resize(num_histogram_bins, 0);
  }

  // build histogram
  for (const auto &filename : wave_filenames) {
    bool is_ok = false;
    std::vector<float> samples =
        sherpa_ncnn::ReadWave(filename, expected_sampling_rate, &is_ok);
    if (!is_ok) {
      fprintf(stderr, "Failed to read %s\n", filename.c_str());
      continue;
    }
    fprintf(stderr, "Processing %s\n", filename.c_str());

    sherpa_ncnn::FeatureExtractorConfig config;
    config.sampling_rate = 16000;
    config.feature_dim = 80;
    sherpa_ncnn::FeatureExtractor feature_extractor(config);
    feature_extractor.AcceptWaveform(expected_sampling_rate, samples.data(),
                                     samples.size());
    feature_extractor.InputFinished();

    int32_t context_size = model->ContextSize();
    int32_t blank_id = model->BlankId();

    std::vector<int32_t> hyp(context_size, blank_id);

    ncnn::Mat decoder_input(context_size);
    for (int32_t i = 0; i != context_size; ++i) {
      static_cast<int32_t *>(decoder_input)[i] = blank_id;
    }

    ncnn::Mat decoder_out = model->RunDecoder(decoder_input);

    std::vector<ncnn::Mat> states;
    ncnn::Mat encoder_out;

    int32_t num_processed = 0;
    while (feature_extractor.NumFramesReady() - num_processed >= segment) {
      ncnn::Extractor encoder_ex = model->GetEncoder().create_extractor();
      encoder_ex.set_light_mode(false);
      encoder_ex.set_blob_allocator(&blob_allocators[0]);
      encoder_ex.set_workspace_allocator(&workspace_allocators[0]);

      ncnn::Extractor joiner_ex = model->GetJoiner().create_extractor();
      joiner_ex.set_light_mode(false);
      joiner_ex.set_blob_allocator(&blob_allocators[0]);
      joiner_ex.set_workspace_allocator(&workspace_allocators[0]);

      ncnn::Mat features = feature_extractor.GetFrames(num_processed, segment);
      num_processed += offset;
      std::tie(encoder_out, states) =
          model->RunEncoder(features, states, &encoder_ex);

      for (int j = 0; j < encoder_conv_bottom_blob_count; j++) {
        ncnn::Mat out;
        encoder_ex.extract(encoder_conv_bottom_blobs[j], out);

        // count histogram bin
        {
          const float absmax = encoder_quant_blob_stats[j].absmax;

          std::vector<uint64_t> histogram(num_histogram_bins, 0);

          const int outc = out.c;
          const int outsize = out.w * out.h;
          for (int p = 0; p < outc; p++) {
            const float *ptr = out.channel(p);
            for (int k = 0; k < outsize; k++) {
              if (ptr[k] == 0.f) continue;

              const int index =
                  std::min((int)(fabs(ptr[k]) / absmax * num_histogram_bins),
                           (num_histogram_bins - 1));

              histogram[index] += 1;
            }
          }

          QuantBlobStat &stat = encoder_quant_blob_stats[j];

          for (int k = 0; k < num_histogram_bins; k++) {
            stat.histogram[k] += histogram[k];
          }
        }
      }  // for (int j = 0; j < encoder_conv_bottom_blob_count; j++)

      // now for joiner
      for (int32_t t = 0; t != encoder_out.h; ++t) {
        ncnn::Mat encoder_out_t(encoder_out.w, encoder_out.row(t));
        ncnn::Mat joiner_out =
            model->RunJoiner(encoder_out_t, decoder_out, &joiner_ex);

        for (int j = 0; j < joiner_conv_bottom_blob_count; j++) {
          ncnn::Mat out;
          joiner_ex.extract(joiner_conv_bottom_blobs[j], out);

          // count histogram bin
          {
            const float absmax = joiner_quant_blob_stats[j].absmax;

            std::vector<uint64_t> histogram(num_histogram_bins, 0);

            const int outc = out.c;
            const int outsize = out.w * out.h;
            for (int p = 0; p < outc; p++) {
              const float *ptr = out.channel(p);
              for (int k = 0; k < outsize; k++) {
                if (ptr[k] == 0.f) continue;

                const int index =
                    std::min((int)(fabs(ptr[k]) / absmax * num_histogram_bins),
                             (num_histogram_bins - 1));

                histogram[index] += 1;
              }
            }

            QuantBlobStat &stat = joiner_quant_blob_stats[j];

            for (int k = 0; k < num_histogram_bins; k++) {
              stat.histogram[k] += histogram[k];
            }
          }
        }  // for (int j = 0; j < joiner_conv_bottom_blob_count; j++)

        auto y = static_cast<int32_t>(std::distance(
            static_cast<const float *>(joiner_out),
            std::max_element(
                static_cast<const float *>(joiner_out),
                static_cast<const float *>(joiner_out) + joiner_out.w)));

        if (y != blank_id) {
          static_cast<int32_t *>(decoder_input)[0] = hyp.back();
          static_cast<int32_t *>(decoder_input)[1] = y;
          hyp.push_back(y);

          decoder_out = model->RunDecoder(decoder_input);
        }
      }  // for (int32_t t = 0; t != encoder_out.h; ++t)

    }  // while (feature_extractor.NumFramesReady() - num_processed >=
       // segment)
  }    // for (const auto &filename : wave_filenames)

  // using kld to find the best threshold value
  for (int i = 0; i < encoder_conv_bottom_blob_count; i++) {
    QuantBlobStat &stat = encoder_quant_blob_stats[i];

    float scale = compute_kl_threshold(stat, num_histogram_bins);

    encoder_bottom_blob_scales[i].create(1);
    encoder_bottom_blob_scales[i][0] = scale;
  }  // for (int i = 0; i < encoder_conv_bottom_blob_count; i++)

  for (int i = 0; i < joiner_conv_bottom_blob_count; i++) {
    QuantBlobStat &stat = joiner_quant_blob_stats[i];

    float scale = compute_kl_threshold(stat, num_histogram_bins);

    joiner_bottom_blob_scales[i].create(1);
    joiner_bottom_blob_scales[i][0] = scale;
  }  // for (int i = 0; i < joiner_conv_bottom_blob_count; i++)

  return 0;
}

void QuantNet::print_quant_info() const {
  fprintf(stderr, "----------encoder----------\n");
  for (int i = 0; i < (int)encoder_conv_bottom_blobs.size(); i++) {
    const QuantBlobStat &stat = encoder_quant_blob_stats[i];

    float scale = 127 / stat.threshold;

    fprintf(stderr, "%-40s : max = %-15f  threshold = %-15f  scale = %-15f\n",
            encoder_layers[encoder_conv_layers[i]]->name.c_str(), stat.absmax,
            stat.threshold, scale);
  }

  fprintf(stderr, "----------joiner----------\n");
  // for joiner
  for (int i = 0; i < (int)joiner_conv_bottom_blobs.size(); i++) {
    const QuantBlobStat &stat = joiner_quant_blob_stats[i];

    float scale = 127 / stat.threshold;

    fprintf(stderr, "%-40s : max = %-15f  threshold = %-15f  scale = %-15f\n",
            joiner_layers[joiner_conv_layers[i]]->name.c_str(), stat.absmax,
            stat.threshold, scale);
  }
}

int QuantNet::save_table_encoder(const char *tablepath) {
  FILE *fp = fopen(tablepath, "wb");
  if (!fp) {
    fprintf(stderr, "fopen %s failed\n", tablepath);
    return -1;
  }

  const int encoder_conv_layer_count = (int)encoder_conv_layers.size();
  const int encoder_conv_bottom_blob_count =
      (int)encoder_conv_bottom_blobs.size();

  for (int i = 0; i < encoder_conv_layer_count; i++) {
    const ncnn::Mat &weight_scale = encoder_weight_scales[i];

    fprintf(fp, "%s_param_0 ",
            encoder_layers[encoder_conv_layers[i]]->name.c_str());
    for (int j = 0; j < weight_scale.w; j++) {
      fprintf(fp, "%f ", weight_scale[j]);
    }
    fprintf(fp, "\n");
  }

  for (int i = 0; i < encoder_conv_bottom_blob_count; i++) {
    const ncnn::Mat &bottom_blob_scale = encoder_bottom_blob_scales[i];

    fprintf(fp, "%s ", encoder_layers[encoder_conv_layers[i]]->name.c_str());
    for (int j = 0; j < bottom_blob_scale.w; j++) {
      fprintf(fp, "%f ", bottom_blob_scale[j]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);

  return 0;
}

int QuantNet::save_table_joiner(const char *tablepath) {
  FILE *fp = fopen(tablepath, "wb");
  if (!fp) {
    fprintf(stderr, "fopen %s failed\n", tablepath);
    return -1;
  }

  const int joiner_conv_layer_count = (int)joiner_conv_layers.size();
  const int joiner_conv_bottom_blob_count =
      (int)joiner_conv_bottom_blobs.size();

  for (int i = 0; i < joiner_conv_layer_count; i++) {
    const ncnn::Mat &weight_scale = joiner_weight_scales[i];

    fprintf(fp, "%s_param_0 ",
            joiner_layers[joiner_conv_layers[i]]->name.c_str());

    for (int j = 0; j < weight_scale.w; j++) {
      fprintf(fp, "%f ", weight_scale[j]);
    }
    fprintf(fp, "\n");
  }

  for (int i = 0; i < joiner_conv_bottom_blob_count; i++) {
    const ncnn::Mat &bottom_blob_scale = joiner_bottom_blob_scales[i];

    fprintf(fp, "%s ", joiner_layers[joiner_conv_layers[i]]->name.c_str());
    for (int j = 0; j < bottom_blob_scale.w; j++) {
      fprintf(fp, "%f ", bottom_blob_scale[j]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);

  fprintf(stderr,
          "ncnn int8 calibration table create success, best wish for your int8 "
          "inference has a low accuracy loss...\\(^0^)/...233...\n");

  return 0;
}

static std::vector<std::string> ReadWaveFilenames(const char *f) {
  std::ifstream in(f);
  std::vector<std::string> ans;
  std::string line;
  while (std::getline(in, line)) {
    ans.push_back(line);
  }
  return ans;
}

static void ShowUsage() {
  fprintf(
      stderr,
      "Usage:\ngenerate-int8-scale-table encoder.param "
      "encoder.bin decoder.param decoder.bin joiner.param joiner.bin "
      "encoder-scale-table.txt joiner-scale-table.txt wave_filenames.txt\n\n"
      "Each line in wave_filenames.txt is a path to some 16k Hz mono wave "
      "file.\n");
}

int main(int argc, char **argv) {
  if (argc != 10) {
    fprintf(stderr, "Please provide 10 arg. Currently given: %d\n", argc);

    ShowUsage();
    return 1;
  }

  int32_t num_threads = 10;
  sherpa_ncnn::ModelConfig config;

  config.encoder_param = argv[1];
  config.encoder_bin = argv[2];
  config.decoder_param = argv[3];
  config.decoder_bin = argv[4];
  config.joiner_param = argv[5];
  config.joiner_bin = argv[6];

  const char *encoder_scale_table = argv[7];
  const char *joiner_scale_table = argv[8];
  std::vector<std::string> wave_filenames = ReadWaveFilenames(argv[9]);

  ncnn::Option opt;
  opt.num_threads = num_threads;
  opt.lightmode = false;
  opt.use_fp16_packed = false;
  opt.use_fp16_storage = false;
  opt.use_fp16_arithmetic = false;

  config.encoder_opt = opt;
  config.decoder_opt = opt;
  config.joiner_opt = opt;

  auto model = sherpa_ncnn::Model::Create(config);

  QuantNet net(model.get());

  net.init();

  // TODO(fangjun): We support only KL right now.
  net.quantize_KL(wave_filenames);

  net.print_quant_info();

  net.save_table_encoder(encoder_scale_table);
  net.save_table_joiner(joiner_scale_table);

  return 0;
}
