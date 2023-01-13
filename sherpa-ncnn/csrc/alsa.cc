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

#include "alsa/asoundlib.h"

namespace sherpa_ncnn {

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

  // 32-bit float rather than SND_PCM_FORMAT_S16_LE
  // so that we can use float samples directly.
  err = snd_pcm_hw_params_set_format(capture_handle_, hw_params,
                                     SND_PCM_FORMAT_FLOAT_LE);
  if (err) {
    fprintf(stderr, "Failed to set sample format: %s\n", snd_strerror(err));
    exit(-1);
  }

  // mono
  err = snd_pcm_hw_params_set_channels(capture_handle_, hw_params, 1);
  if (err) {
    fprintf(stderr, "Failed to set number of channels to 1. %s\n",
            snd_strerror(err));
    exit(-1);
  }

  unsigned int32_t actual_sample_rate = expected_sample_rate_;

  unsigned int sample_rate = 16000;
  int dir = 0;
  err = snd_pcm_hw_params_set_rate_near(capture_handle_, hw_params,
                                        &actual_sample_rate, &dir);
  if (err) {
    fprintf(stderr, "Failed to set sample rate to, %u: %s\n", sample_rate,
            snd_strerror(err));
    exit(-1);
  }
  actual_sample_rate_ = actual_sample_rate;

  if (actual_sample_rate_ != expected_sample_rate_) {
    fprintf(stderr, "Failed to set sample rate to %d\n", expected_sample_rate_);
    fprintf(stderr, "Current sample rate to %d\n", actual_sample_rate_);
    fprintf(stderr,
            "Creating a resampler:\n"
            "   in_sample_rate: %d\n"
            "   output_sample_rate: %d\n",
            actual_sample_rate_, expected_sample_rate_);

    float min_freq = std::min(actual_sample_rate_, expected_sample_rate_);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    resampler_ = std::make_unqiue<LinearResample>(
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

  err = snd_pcm_prepare(capture_handle);
  if (err) {
    fprintf(stderr, "Failed to prepare for recording: %s\n", snd_strerror(err));
    exit(-1);
  }

  fprintf(stderr, "Recording started!\n");
}

Alsa::~Alsa() { snd_pcm_close(capture_handle_); }

}  // namespace sherpa_ncnn

#endif
