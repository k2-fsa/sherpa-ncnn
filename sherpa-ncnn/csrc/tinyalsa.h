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

#ifndef SHERPA_NCNN_CSRC_TINYALSA_H_
#define SHERPA_NCNN_CSRC_TINYALSA_H_

#include <memory>
#include <vector>

#include "sherpa-ncnn/csrc/resample.h"
#include "tinyalsa/asoundlib.h"

namespace sherpa_ncnn {

class TinyAlsa {
 public:
  explicit TinyAlsa(const char *device_name);
  ~TinyAlsa();

  // This is a blocking read.
  //
  // @param num_samples  Number of samples to read.
  //
  // The returned value is valid until the next call to Read().
  const std::vector<float> &Read();

  int32_t GetExpectedSampleRate() const { return expected_sample_rate_; }
  int32_t GetActualSampleRate() const { return pcm_config_.rate; }

 private:
  struct pcm *tinyalsa_pcm_ = nullptr;
  int32_t expected_sample_rate_ = 16000;

  struct pcm_config pcm_config_;

  std::unique_ptr<LinearResample> resampler_;
  std::vector<char> samples_;  // directly from the microphone

  // normalized version of samples_
  std::vector<float> samples1_;
  std::vector<float> samples2_;  // possibly resampled from samples1_
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_TINYALSA_H_
