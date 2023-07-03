/**
 * Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
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

// #ifdef SHERPA_NCNN_ENABLE_TINYALSA

#include "sherpa-ncnn/csrc/tinyalsa.h"

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <limits>

#include "tinyalsa/asoundlib.h"

// see https://www.linuxjournal.com/article/6735
// buffer: a list of periods
// period: a list of frames
// frame: LR (left channel sample + right channel sample)
// sample:

namespace sherpa_ncnn {

static pcm_config GetPcmConfig(uint32_t card, uint32_t device, uint32_t flags) {
  struct pcm_config config;
  memset(&config, 0, sizeof(config));

  struct pcm_params *params = pcm_params_get(card, device, flags);
  std::vector<char> s(1024);
  pcm_params_to_string(params, s.data(), s.size() - 1);
  fprintf(stderr, "%s\n", s.data());

  // we test only 3 formats: PCM_FORMAT_S16_LE, PCM_FORMAT_S32_LE,
  // PCM_FORMAT_FLOAT_LE.
  // If you want to support more formats, please either create an issue
  // or change the code by yourself.
  std::vector<std::pair<const char *, enum pcm_format>> supported_formats = {
      {"PCM_FORMAT_S16_LE", PCM_FORMAT_S16_LE},
      {"PCM_FORMAT_S32_LE", PCM_FORMAT_S32_LE},
      {"PCM_FORMAT_FLOAT_LE", PCM_FORMAT_FLOAT_LE},
  };

  config.format = PCM_FORMAT_INVALID;
  for (const auto &p : supported_formats) {
    fprintf(stderr, "Testing %s\n", p.first);

    if (pcm_params_format_test(params, p.second)) {
      fprintf(stderr, "%s is supported!\n", p.first);
      config.format = PCM_FORMAT_S16_LE;
      break;
    } else {
      fprintf(stderr, "  %s is NOT supported!\n", p.first);
    }
  }

  if (config.format == PCM_FORMAT_INVALID) {
    fprintf(stderr, "Failed to find the supported PCM format\n");
    exit(-1);
  }

  // now for channels
  uint32_t min_channel = pcm_params_get_min(params, PCM_PARAM_CHANNELS);
  uint32_t max_channel = pcm_params_get_max(params, PCM_PARAM_CHANNELS);
  if (min_channel != max_channel) {
    fprintf(stderr, "Minimum channel count: %u, maximum channel count: %u\n",
            min_channel, max_channel);
  }
  fprintf(stderr, "Number of channels to use: %u\n", min_channel);
  config.channels = min_channel;

  // for sample rate
  uint32_t min_rate = pcm_params_get_min(params, PCM_PARAM_RATE);
  uint32_t max_rate = pcm_params_get_max(params, PCM_PARAM_RATE);

  if (min_rate != max_rate) {
    fprintf(stderr, "Minimum sample rate: %u, maximum sample rate: %u\n",
            min_rate, max_rate);
  }
  fprintf(stderr, "Sample rate to use: %u\n", min_rate);
  config.rate = min_rate;

  // for periods
  uint32_t min_period_size = pcm_params_get_min(params, PCM_PARAM_PERIOD_SIZE);
  uint32_t max_period_size = pcm_params_get_max(params, PCM_PARAM_PERIOD_SIZE);
  fprintf(stderr, "min period size: %u, max period size: %u\n", min_period_size,
          max_period_size);

  // 0.1 second
  uint32_t selected_period_size = config.rate * 0.1;

  if (selected_period_size >= min_period_size &&
      selected_period_size <= max_period_size) {
    config.period_size = selected_period_size;
  } else {
    config.period_size = min_period_size;
  }

  fprintf(stderr, "Use period size: %u\n", config.period_size);
  config.period_count = 10;

  pcm_params_free(params);

  return config;
}

template <typename T>
void ToFloat(const std::vector<char> &in, int32_t num_channels,
             std::vector<float> *out) {
  out->resize(in.size() / num_channels);

  auto p = reinterpret_cast<const T *>(in.data());
  int32_t num_samples = in.size() / sizeof(T);
  fprintf(stderr, "num_samples: %d, num_channels: %d\n", num_samples,
          num_channels);

  for (int32_t i = 0, k = 0; i < num_samples; i += num_channels, ++k) {
    (*out)[k] = p[i] / float(std::numeric_limits<T>::max());
  }
}

void ToFloat(const std::vector<char> &in, int32_t num_channels,
             std::vector<float> *out) {
  out->resize(in.size() / num_channels);

  auto p = reinterpret_cast<const float *>(in.data());
  int32_t num_samples = in.size() / sizeof(float);

  for (int32_t i = 0, k = 0; i < num_samples; i += num_channels, ++k) {
    (*out)[k] = p[i];
  }
}

TinyAlsa::TinyAlsa(const char *device_name) {
  fprintf(stderr, "device_name: %s\n", device_name);

  if (device_name[0] != 'h' || device_name[1] != 'w' || device_name[2] != ':') {
    fprintf(stderr, "Invalid device name format: %s\n", device_name);
    exit(-1);
  }

  unsigned int card;
  unsigned int device;
  if (sscanf(&device_name[3], "%u,%u", &card, &device) != 2) {
    fprintf(stderr, "Failed to get card and device from %s\n", device_name);
    exit(-1);
  }
  fprintf(stderr, "card: %u, device: %u\n", card, device);

  int flags = PCM_IN;
  pcm_config_ = GetPcmConfig(card, device, flags);

  tinyalsa_pcm_ = pcm_open(card, device, flags, &pcm_config_);
  if (tinyalsa_pcm_ == nullptr) {
    fprintf(stderr, "Failed to allocate memory for PCM\n");
    exit(-1);
  }

  if (!pcm_is_ready(tinyalsa_pcm_)) {
    pcm_close(tinyalsa_pcm_);
    tinyalsa_pcm_ = nullptr;

    fprintf(stderr, "Failed to open PCM: %s.\n", pcm_get_error(tinyalsa_pcm_));
    exit(-1);
  }

  int32_t actual_sample_rate = pcm_config_.rate;

  if (actual_sample_rate != expected_sample_rate_) {
    fprintf(stderr, "Failed to set sample rate to %d\n", expected_sample_rate_);
    fprintf(stderr, "Current sample rate is %d\n", actual_sample_rate);
    fprintf(stderr,
            "Creating a resampler:\n"
            "   in_sample_rate: %d\n"
            "   output_sample_rate: %d\n",
            actual_sample_rate, expected_sample_rate_);

    float min_freq = std::min(actual_sample_rate, expected_sample_rate_);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    resampler_ = std::make_unique<LinearResample>(
        actual_sample_rate, expected_sample_rate_, lowpass_cutoff,
        lowpass_filter_width);
  }
}

TinyAlsa::~TinyAlsa() {
  if (tinyalsa_pcm_) {
    pcm_close(tinyalsa_pcm_);
  }
}

const std::vector<float> &TinyAlsa::Read() {
  // buffer_size: byte_per_sample * num_channels * period_size
  uint32_t buffer_size =
      pcm_frames_to_bytes(tinyalsa_pcm_, pcm_config_.period_size);
  samples_.resize(buffer_size);

  // a frame contain samples from multiple channels at time t
  uint32_t num_frames = pcm_bytes_to_frames(tinyalsa_pcm_, buffer_size);

  // count is in frames. Each frame contains pcm_config_.channels samples
  int32_t count = pcm_readi(tinyalsa_pcm_, samples_.data(), num_frames);
  if (count < 0) {
    fprintf(stderr, "Read samples failed. count: %d\n", count);
    exit(-1);
  }
  fprintf(stderr, "read count: %d, buffer_size: %u\n", count, buffer_size);

  if (pcm_config_.format == PCM_FORMAT_S16_LE) {
    samples_.resize(count * pcm_config_.channels * sizeof(int16_t));
    ToFloat<int16_t>(samples_, pcm_config_.channels, &samples1_);
  } else if (pcm_config_.format == PCM_FORMAT_S32_LE) {
    samples_.resize(count * pcm_config_.channels * sizeof(int32_t));
    ToFloat<int32_t>(samples_, pcm_config_.channels, &samples1_);
  } else if (pcm_config_.format == PCM_FORMAT_FLOAT_LE) {
    samples_.resize(count * pcm_config_.channels * sizeof(float));
    ToFloat(samples_, pcm_config_.channels, &samples1_);
  } else {
  }

  if (!resampler_) {
    return samples1_;
  }

  resampler_->Resample(samples1_.data(), samples_.size(), false, &samples2_);
  return samples2_;
}

}  // namespace sherpa_ncnn
