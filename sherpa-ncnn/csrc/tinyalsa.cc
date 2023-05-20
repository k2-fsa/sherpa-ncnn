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

#include <algorithm>
#include <string.h>

#include "tinyalsa/asoundlib.h"

namespace sherpa_ncnn {

void ToFloat(const std::vector<int16_t> &in, int32_t num_channels,
             std::vector<float> *out) {
  out->resize(in.size() / num_channels);

  int32_t n = in.size();
  for (int32_t i = 0, k = 0; i < n; i += num_channels, ++k) {
    (*out)[k] = in[i] / 32768.;
  }
}

TinyAlsa::TinyAlsa(const char *device_name) {
  unsigned int card = 0;
  unsigned int device = 0;
  int flags = PCM_IN;

  struct pcm_config config;

  memset(&config, 0, sizeof(config));
  config.channels = 2;
  config.rate = 48000;
  config.format = PCM_FORMAT_S32_LE;
  config.period_size = 1024;
  config.period_count = 2;
  config.start_threshold = 1024;
  config.silence_threshold = 1024 * 2;
  config.stop_threshold = 1024 * 2;

  tinyalsa_pcm  = pcm_open(card, device, flags, &config);
  if (tinyalsa_pcm == NULL) {
      fprintf(stderr, "failed to allocate memory for PCM\n");
  } else if (!pcm_is_ready(tinyalsa_pcm)) {
      pcm_close(tinyalsa_pcm);
      fprintf(stderr, "failed to open PCM\n");
  }
}

TinyAlsa::~TinyAlsa() { pcm_close(tinyalsa_pcm); }

const std::vector<float> &TinyAlsa::Read(int32_t num_samples) {
  samples_.resize(num_samples * actual_channel_count_);

  // count is in frames. Each frame contains actual_channel_count_ samples
  int32_t count = pcm_readi(tinyalsa_pcm, samples_.data(), num_samples);

  samples_.resize(count * actual_channel_count_);

  ToFloat(samples_, actual_channel_count_, &samples1_);

  if (!resampler_) {
    return samples1_;
  }

  resampler_->Resample(samples1_.data(), samples_.size(), false, &samples2_);
  return samples2_;
}

}  // namespace sherpa_ncnn

// #endif
