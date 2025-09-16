/**
 * Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)
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

#ifndef SHERPA_NCNN_CSRC_SILERO_VAD_MODEL_CONFIG_H_
#define SHERPA_NCNN_CSRC_SILERO_VAD_MODEL_CONFIG_H_

#include <memory>
#include <string>

#include "sherpa-ncnn/csrc/parse-options.h"

namespace sherpa_ncnn {

struct SileroVadModelConfig {
  // It should contain silero.ncnn.param and silero.ncnn.bin
  std::string model_dir;

  // threshold to classify a segment as speech
  //
  // If the predicted probability of a segment is larger than this
  // value, then it is classified as speech.
  float threshold = 0.5;

  float min_silence_duration = 0.5;  // in seconds

  float min_speech_duration = 0.25;  // in seconds

  // 512, 1024, 1536 samples for 16000 Hz
  // 256, 512, 768 samples for 800 Hz
  int32_t window_size = 512;  // in samples

  int32_t sample_rate = 16000;

  bool use_vulkan_compute = true;
  int32_t num_threads = 1;

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_SILERO_VAD_MODEL_CONFIG_H_
