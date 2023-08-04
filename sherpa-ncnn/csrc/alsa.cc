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

#ifdef SHERPA_NCNN_ENABLE_ALSA

#include "sherpa-ncnn/csrc/alsa.h"

#include <algorithm>
#include <cstdlib>

#include "alsa/asoundlib.h"

namespace sherpa_ncnn {

void ToFloat16(const std::vector<int16_t> &in, int32_t channel_to_use,
               int32_t num_channels, std::vector<float> *out) {
  out->resize(in.size() / num_channels);

  int32_t n = in.size();
  for (int32_t i = 0, k = 0; i < n; i += num_channels, ++k) {
    (*out)[k] = in[i + channel_to_use] / 32768.0;
  }
}

void ToFloat32(const std::vector<int32_t> &in, int32_t channel_to_use,
               int32_t num_channels, std::vector<float> *out) {
  out->resize(in.size() / num_channels);

  int32_t n = in.size();
  for (int32_t i = 0, k = 0; i < n; i += num_channels, ++k) {
    (*out)[k] = in[i + channel_to_use] / static_cast<float>(1 << 31);
  }
}

Alsa::Alsa(const char *device_name) {
  const char *kDeviceHelp = R"(
Please use the command:

  arecord -l

to list all available devices. For instance, if the output is:

**** List of CAPTURE Hardware Devices ****
card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

and if you want to select card 3 and the device 0 on that card, please use:

  hw:3,0

  )";

  int32_t err =
      snd_pcm_open(&capture_handle_, device_name, SND_PCM_STREAM_CAPTURE, 0);
  if (err) {
    fprintf(stderr, "Unable to open: %s. %s\n", device_name, snd_strerror(err));
    fprintf(stderr, "%s\n", kDeviceHelp);
    exit(-1);
  }

  snd_pcm_hw_params_t *hw_params;
  snd_pcm_hw_params_alloca(&hw_params);

  err = snd_pcm_hw_params_any(capture_handle_, hw_params);
  if (err) {
    fprintf(stderr, "Failed to initialize hw_params: %s\n", snd_strerror(err));
    exit(-1);
  }

  err = snd_pcm_hw_params_set_access(capture_handle_, hw_params,
                                     SND_PCM_ACCESS_RW_INTERLEAVED);
  if (err) {
    fprintf(stderr, "Failed to set access type: %s\n", snd_strerror(err));
    exit(-1);
  }

  err = snd_pcm_hw_params_set_format(capture_handle_, hw_params,
                                     SND_PCM_FORMAT_S16_LE);
  if (err) {
    fprintf(stderr, "Failed to set format to SND_PCM_FORMAT_S16_LE: %s\n",
            snd_strerror(err));

    // now try to use SND_PCM_FORMAT_S32_LE
    fprintf(stderr, "Trying to set format to SND_PCM_FORMAT_S32_LE\n");

    err = snd_pcm_hw_params_set_format(capture_handle_, hw_params,
                                       SND_PCM_FORMAT_S32_LE);
    if (err) {
      fprintf(stderr, "Failed to set format to SND_PCM_FORMAT_S32_LE: %s\n",
              snd_strerror(err));
      exit(-1);
    }
    fprintf(stderr, "Set format to SND_PCM_FORMAT_S32_LE successfully\n");
    pcm_format_ = 32;
  }

  std::vector<int32_t> possible_channels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  actual_channel_count_ = -1;
  for (auto i : possible_channels) {
    err = snd_pcm_hw_params_set_channels(capture_handle_, hw_params, i);
    if (err) {
      fprintf(stderr, "Failed to set number of channels to %d. %s\n", i,
              snd_strerror(err));
    } else {
      actual_channel_count_ = i;
      break;
    }
  }

  if (actual_channel_count_ == -1) {
    fprintf(stderr, "Please replace your microphone!\n");
    exit(-1);
  }

  if (actual_channel_count_ > 1) {
    const char *p = std::getenv("SHERPA_NCNN_ALSA_USE_CHANNEL");
    if (p != nullptr) {
      int32_t channel_to_use = atoi(p);
      if (channel_to_use < 0 || channel_to_use >= actual_channel_count_) {
        fprintf(stderr, "Invalid SHERPA_NCNN_ALSA_USE_CHANNEL: %s\n", p);
        exit(-1);
      }

      channel_to_use_ = channel_to_use;
    }

    fprintf(stderr, "We use only channel %d out of %d channels\n",
            channel_to_use_, actual_channel_count_);

    fprintf(stderr,
            "Please use arecord and audacity to check that channel %d indeed "
            "contains audio samples\n",
            channel_to_use_);
    fprintf(stderr,
            "Hint: You can use\n"
            "  export SHERPA_NCNN_ALSA_USE_CHANNEL=1\n"
            "to use channel 1 out of %d channels\n",
            actual_channel_count_);
  }

  uint32_t actual_sample_rate = expected_sample_rate_;

  int32_t dir = 0;
  err = snd_pcm_hw_params_set_rate_near(capture_handle_, hw_params,
                                        &actual_sample_rate, &dir);
  if (err) {
    fprintf(stderr, "Failed to set sample rate to, %d: %s\n",
            expected_sample_rate_, snd_strerror(err));
    exit(-1);
  }
  actual_sample_rate_ = actual_sample_rate;

  if (actual_sample_rate_ != expected_sample_rate_) {
    fprintf(stderr, "Failed to set sample rate to %d\n", expected_sample_rate_);
    fprintf(stderr, "Current sample rate is %d\n", actual_sample_rate_);
    fprintf(stderr,
            "Creating a resampler:\n"
            "   in_sample_rate: %d\n"
            "   output_sample_rate: %d\n",
            actual_sample_rate_, expected_sample_rate_);

    float min_freq = std::min(actual_sample_rate_, expected_sample_rate_);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    resampler_ = std::make_unique<LinearResample>(
        actual_sample_rate_, expected_sample_rate_, lowpass_cutoff,
        lowpass_filter_width);
  } else {
    fprintf(stderr, "Current sample rate: %d\n", actual_sample_rate_);
  }

  err = snd_pcm_hw_params(capture_handle_, hw_params);
  if (err) {
    fprintf(stderr, "Failed to set hw params: %s\n", snd_strerror(err));
    exit(-1);
  }

  err = snd_pcm_prepare(capture_handle_);
  if (err) {
    fprintf(stderr, "Failed to prepare for recording: %s\n", snd_strerror(err));
    exit(-1);
  }

  fprintf(stderr, "Recording started!\n");
}

Alsa::~Alsa() { snd_pcm_close(capture_handle_); }

const std::vector<float> &Alsa::Read16(int32_t num_samples) {
  samples16_.resize(num_samples * actual_channel_count_);

  // count is in frames. Each frame contains actual_channel_count_ samples
  int32_t count =
      snd_pcm_readi(capture_handle_, samples16_.data(), num_samples);
  if (count == -EPIPE) {
    fprintf(
        stderr,
        "An overrun occurred, which means the RTF of the current "
        "model on your board is larger than 1. You can use ./bin/sherpa-ncnn "
        "to verify that. Please select a smaller model whose RTF is less than "
        "1 for your board.");
    exit(-1);
  }

  samples16_.resize(count * actual_channel_count_);

  ToFloat16(samples16_, channel_to_use_, actual_channel_count_, &samples1_);

  if (!resampler_) {
    return samples1_;
  }

  resampler_->Resample(samples1_.data(), samples16_.size(), false, &samples2_);
  return samples2_;
}

const std::vector<float> &Alsa::Read32(int32_t num_samples) {
  samples32_.resize(num_samples * actual_channel_count_);

  // count is in frames. Each frame contains actual_channel_count_ samples
  int32_t count =
      snd_pcm_readi(capture_handle_, samples32_.data(), num_samples);
  if (count == -EPIPE) {
    fprintf(
        stderr,
        "An overrun occurred, which means the RTF of the current "
        "model on your board is larger than 1. You can use ./bin/sherpa-ncnn "
        "to verify that. Please select a smaller model whose RTF is less than "
        "1 for your board.");
    exit(-1);
  }

  samples32_.resize(count * actual_channel_count_);

  ToFloat32(samples32_, channel_to_use_, actual_channel_count_, &samples1_);

  if (!resampler_) {
    return samples1_;
  }

  resampler_->Resample(samples1_.data(), samples32_.size(), false, &samples2_);
  return samples2_;
}

const std::vector<float> &Alsa::Read(int32_t num_samples) {
  switch (pcm_format_) {
    case 16:
      return Read16(num_samples);
    case 32:
      return Read32(num_samples);
    default:
      fprintf(stderr, "Unsupported pcm format: %d\n", pcm_format_);
      exit(-1);
  }
}

}  // namespace sherpa_ncnn

#endif
