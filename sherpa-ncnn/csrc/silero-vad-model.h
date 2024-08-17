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

#ifndef SHERPA_NCNN_CSRC_SILERO_VAD_MODEL_H_
#define SHERPA_NCNN_CSRC_SILERO_VAD_MODEL_H_

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include <memory>

#include "sherpa-ncnn/csrc/silero-vad-model-config.h"

namespace sherpa_ncnn {

class SileroVadModel {
 public:
  explicit SileroVadModel(const SileroVadModelConfig &config);

#if __ANDROID_API__ >= 9
  SileroVadModel(AAssetManager *mgr, const SileroVadModelConfig &config);
#endif

  ~SileroVadModel();

  // reset the internal model states
  void Reset();

  /**
   * @param samples Pointer to a 1-d array containing audio samples.
   *                Each sample should be normalized to the range [-1, 1].
   * @param n Number of samples.
   *
   * @return Return true if speech is detected. Return false otherwise.
   */
  bool IsSpeech(const float *samples, int32_t n);

  // For silero vad V4, it is WindowShift().
  // For silero vad V5, it is WindowShift()+64 for 16kHz and
  //                          WindowShift()+32 for 8kHz
  int32_t WindowSize() const;

  // 512
  int32_t WindowShift() const;

  int32_t MinSilenceDurationSamples() const;
  int32_t MinSpeechDurationSamples() const;

  void SetMinSilenceDuration(float s);
  void SetThreshold(float threshold);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_SILERO_VAD_MODEL_H_
