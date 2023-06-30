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

#ifndef SHERPA_NCNN_CSRC_ALSA_H_
#define SHERPA_NCNN_CSRC_ALSA_H_

#include <memory>
#include <vector>

#include "alsa/asoundlib.h"
#include "sherpa-ncnn/csrc/resample.h"

namespace sherpa_ncnn {

class Alsa {
 public:
  explicit Alsa(const char *device_name);
  ~Alsa();

  // This is a blocking read.
  //
  // @param num_samples  Number of samples to read.
  //
  // The returned value is valid until the next call to Read().
  const std::vector<float> &Read(int32_t num_samples);

  int32_t GetExpectedSampleRate() const { return expected_sample_rate_; }
  int32_t GetActualSampleRate() const { return actual_sample_rate_; }

 private:
  const std::vector<float> &Read16(int32_t num_samples);
  const std::vector<float> &Read32(int32_t num_samples);

 private:
  snd_pcm_t *capture_handle_;
  int32_t expected_sample_rate_ = 16000;
  int32_t actual_sample_rate_;

  int32_t actual_channel_count_ = 1;

  // If there are multipel channels, we use this channel for recognition
  int32_t channel_to_use_ = 0;

  std::unique_ptr<LinearResample> resampler_;

  // If it is 16, we use samples16_
  // If it is 32, we use samples32_
  //
  // It can only be 16 or 32.
  int32_t pcm_format_ = 16;

  std::vector<int16_t> samples16_;  // directly from the microphone
  std::vector<int32_t> samples32_;  // directly from the microphone

  std::vector<float> samples1_;  // normalized version of samples_
  std::vector<float> samples2_;  // possibly resampled from samples1_
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_ALSA_H_
