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

#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "layer/convolution.h"
#include "layer/convolutiondepthwise.h"
#include "layer/innerproduct.h"
#include "mat.h"
#include "net.h"
#include "sherpa-ncnn/csrc/features.h"
#include "sherpa-ncnn/csrc/model.h"
#include "sherpa-ncnn/csrc/wave-reader.h"

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

class QuantNet : public ncnn::Net {
 public:
  QuantNet(sherpa_ncnn::Model *model);

  sherpa_ncnn::Model *model;
  std::vector<ncnn::Blob> &blobs;
  std::vector<ncnn::Layer *> &layers;

 public:
  int init();
  void print_quant_info() const;
  int save_table(const char *tablepath);
  int quantize_KL();
  int quantize_ACIQ();
  int quantize_EQ();

 public:
  std::vector<int> conv_layers;
  std::vector<int> conv_bottom_blobs;
  std::vector<int> conv_top_blobs;

  // result
  std::vector<QuantBlobStat> quant_blob_stats;
  std::vector<ncnn::Mat> weight_scales;
  std::vector<ncnn::Mat> bottom_blob_scales;
};

QuantNet::QuantNet(sherpa_ncnn::Model *model)
    : model(model),
      blobs(model->GetEncoder().mutable_blobs()),
      layers(model->GetEncoder().mutable_layers()) {}

int QuantNet::init() {
  // find all conv layers
  for (int i = 0; i < (int)layers.size(); i++) {
    const ncnn::Layer *layer = layers[i];
    if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise" ||
        layer->type == "InnerProduct") {
      conv_layers.push_back(i);
      conv_bottom_blobs.push_back(layer->bottoms[0]);
      conv_top_blobs.push_back(layer->tops[0]);
    }
  }

  fprintf(stderr, "num conv layers: %d\n",
          static_cast<int32_t>(conv_layers.size()));

  const int conv_layer_count = (int)conv_layers.size();
  const int conv_bottom_blob_count = (int)conv_bottom_blobs.size();

  quant_blob_stats.resize(conv_bottom_blob_count);
  weight_scales.resize(conv_layer_count);
  bottom_blob_scales.resize(conv_bottom_blob_count);

  return 0;
}

static float compute_kl_divergence(const std::vector<float> &a,
                                   const std::vector<float> &b) {
  const size_t length = a.size();

  float result = 0;
  for (size_t i = 0; i < length; i++) {
    result += a[i] * log(a[i] / b[i]);
  }

  return result;
}

int QuantNet::quantize_KL() {
  const int conv_layer_count = (int)conv_layers.size();
  const int conv_bottom_blob_count = (int)conv_bottom_blobs.size();

  std::vector<std::string> wave_filenames = {
      "/star-fj/fangjun/open-source/sherpa-ncnn/build/0.wav",
      "/star-fj/fangjun/open-source/sherpa-ncnn/build/1.wav",
      "/star-fj/fangjun/open-source/sherpa-ncnn/build/2.wav",
      "/star-fj/fangjun/open-source/sherpa-ncnn/build/3.wav",
      "/star-fj/fangjun/open-source/sherpa-ncnn/build/4.wav",
  };

  fprintf(stderr, "num files: %d\n", (int)wave_filenames.size());

  const int num_histogram_bins = 2048;
  std::vector<ncnn::UnlockedPoolAllocator> blob_allocators(1);
  std::vector<ncnn::UnlockedPoolAllocator> workspace_allocators(1);

  // initialize conv weight scales
  for (int i = 0; i < conv_layer_count; i++) {
    const ncnn::Layer *layer = layers[conv_layers[i]];
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

      weight_scales[i].create(num_output);
      for (int n = 0; n < num_output; n++) {
        const ncnn::Mat weight_data_n = convolution->weight_data.range(
            weight_data_size_output * n, weight_data_size_output);

        float absmax = 0.f;
        for (int k = 0; k < weight_data_size_output; k++) {
          absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
        }

        if (quant_6bit) {
          weight_scales[i][n] = 31 / absmax;
        } else {
          weight_scales[i][n] = 127 / absmax;
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

      weight_scales[i].create(group);

      for (int n = 0; n < group; n++) {
        const ncnn::Mat weight_data_n = convolutiondepthwise->weight_data.range(
            weight_data_size_output * n, weight_data_size_output);

        float absmax = 0.f;
        for (int k = 0; k < weight_data_size_output; k++) {
          absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
        }

        weight_scales[i][n] = 127 / absmax;
      }
    }  // if (layer->type == "ConvolutionDepthWise")

    if (layer->type == "InnerProduct") {
      const ncnn::InnerProduct *innerproduct =
          (const ncnn::InnerProduct *)layer;

      const int num_output = innerproduct->num_output;
      const int weight_data_size_output =
          innerproduct->weight_data_size / num_output;

      weight_scales[i].create(num_output);

      for (int n = 0; n < num_output; n++) {
        const ncnn::Mat weight_data_n = innerproduct->weight_data.range(
            weight_data_size_output * n, weight_data_size_output);

        float absmax = 0.f;
        for (int k = 0; k < weight_data_size_output; k++) {
          absmax = std::max(absmax, (float)fabs(weight_data_n[k]));
        }

        weight_scales[i][n] = 127 / absmax;
      }
    }  // if (layer->type == "InnerProduct")
  }    // for (int i = 0; i < conv_layer_count; i++)

  float expected_sampling_rate = 16000;

  knf::FbankOptions fbank_opts;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.snip_edges = false;
  fbank_opts.frame_opts.samp_freq = expected_sampling_rate;
  fbank_opts.mel_opts.num_bins = 80;

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

    sherpa_ncnn::FeatureExtractor feature_extractor(fbank_opts);
    feature_extractor.AcceptWaveform(expected_sampling_rate, samples.data(),
                                     samples.size());
    feature_extractor.InputFinished();

    std::vector<ncnn::Mat> states;
    ncnn::Mat encoder_out;

    int32_t num_processed = 0;
    while (feature_extractor.NumFramesReady() - num_processed >= segment) {
      fprintf(stderr, "%d, %d\n", num_processed,
              feature_extractor.NumFramesReady());
      ncnn::Extractor ex = model->GetEncoder().create_extractor();
      ex.set_light_mode(false);
      ex.set_blob_allocator(&blob_allocators[0]);
      ex.set_workspace_allocator(&workspace_allocators[0]);

      ncnn::Mat features = feature_extractor.GetFrames(num_processed, segment);
      num_processed += offset;
      std::tie(encoder_out, states) = model->RunEncoder(features, states, &ex);

      for (int j = 0; j < conv_bottom_blob_count; j++) {
        ncnn::Mat out;
        ex.extract(conv_bottom_blobs[j], out);

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

          QuantBlobStat &stat = quant_blob_stats[j];
          stat.absmax = std::max(stat.absmax, absmax);
        }
      }  // for (int j = 0; j < conv_bottom_blob_count; j++)
    }    // while (feature_extractor.NumFramesReady() - num_processed >=
         // segment)
  }      // for (const auto &filename : wave_filenames)

  // initialize histogram
  for (int i = 0; i < conv_bottom_blob_count; i++) {
    QuantBlobStat &stat = quant_blob_stats[i];

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

    sherpa_ncnn::FeatureExtractor feature_extractor(fbank_opts);
    feature_extractor.AcceptWaveform(expected_sampling_rate, samples.data(),
                                     samples.size());
    feature_extractor.InputFinished();

    std::vector<ncnn::Mat> states;
    ncnn::Mat encoder_out;

    int32_t num_processed = 0;
    while (feature_extractor.NumFramesReady() - num_processed >= segment) {
      fprintf(stderr, "%d, %d\n", num_processed,
              feature_extractor.NumFramesReady());
      ncnn::Extractor ex = model->GetEncoder().create_extractor();
      ex.set_light_mode(false);
      ex.set_blob_allocator(&blob_allocators[0]);
      ex.set_workspace_allocator(&workspace_allocators[0]);

      ncnn::Mat features = feature_extractor.GetFrames(num_processed, segment);
      num_processed += offset;
      std::tie(encoder_out, states) = model->RunEncoder(features, states, &ex);

      for (int j = 0; j < conv_bottom_blob_count; j++) {
        ncnn::Mat out;
        ex.extract(conv_bottom_blobs[j], out);

        // count histogram bin
        {
          const float absmax = quant_blob_stats[j].absmax;

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

          QuantBlobStat &stat = quant_blob_stats[j];

          for (int k = 0; k < num_histogram_bins; k++) {
            stat.histogram[k] += histogram[k];
          }
        }
      }  // for (int j = 0; j < conv_bottom_blob_count; j++)
    }    // while (feature_extractor.NumFramesReady() - num_processed >=
         // segment)
  }      // for (const auto &filename : wave_filenames)

  // using kld to find the best threshold value
  for (int i = 0; i < conv_bottom_blob_count; i++) {
    QuantBlobStat &stat = quant_blob_stats[i];

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

    stat.threshold =
        (target_threshold + 0.5f) * stat.absmax / num_histogram_bins;
    float scale = 127 / stat.threshold;

    bottom_blob_scales[i].create(1);
    bottom_blob_scales[i][0] = scale;
  }  // for (int i = 0; i < conv_bottom_blob_count; i++)
}

void QuantNet::print_quant_info() const {
  for (int i = 0; i < (int)conv_bottom_blobs.size(); i++) {
    const QuantBlobStat &stat = quant_blob_stats[i];

    float scale = 127 / stat.threshold;

    fprintf(stderr, "%-40s : max = %-15f  threshold = %-15f  scale = %-15f\n",
            layers[conv_layers[i]]->name.c_str(), stat.absmax, stat.threshold,
            scale);
  }
}

int QuantNet::save_table(const char *tablepath) {
  FILE *fp = fopen(tablepath, "wb");
  if (!fp) {
    fprintf(stderr, "fopen %s failed\n", tablepath);
    return -1;
  }

  const int conv_layer_count = (int)conv_layers.size();
  const int conv_bottom_blob_count = (int)conv_bottom_blobs.size();

  for (int i = 0; i < conv_layer_count; i++) {
    const ncnn::Mat &weight_scale = weight_scales[i];

    fprintf(fp, "%s_param_0 ", layers[conv_layers[i]]->name.c_str());
    for (int j = 0; j < weight_scale.w; j++) {
      fprintf(fp, "%f ", weight_scale[j]);
    }
    fprintf(fp, "\n");
  }

  for (int i = 0; i < conv_bottom_blob_count; i++) {
    const ncnn::Mat &bottom_blob_scale = bottom_blob_scales[i];

    fprintf(fp, "%s ", layers[conv_layers[i]]->name.c_str());
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

static void ShowUsage() {
  fprintf(stderr,
          "Usage:\ngenerate-int8-scale-table encoder.param "
          "encoder.bin decoder.param decoder.bin joiner.param joiner.bin "
          "scale-table.txt \n");
}

int main(int argc, char **argv) {
  if (argc != 8) {
    fprintf(stderr, "Please provide 8 arg. Currently given: %d\n", argc);

    ShowUsage();
    return 1;
  }

  sherpa_ncnn::ModelConfig config;

  config.encoder_param = argv[1];
  config.encoder_bin = argv[2];
  config.decoder_param = argv[3];
  config.decoder_bin = argv[4];
  config.joiner_param = argv[5];
  config.joiner_bin = argv[6];

  const char *scale_table = argv[7];

  ncnn::Option opt;
  opt.num_threads = 10;
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
  net.quantize_KL();

  net.print_quant_info();

  net.save_table(scale_table);

  return 0;
}
