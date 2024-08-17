/**
 * Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)
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
#ifndef SHERPA_NCNN_CSRC_VOICE_ACTIVITY_DETECTOR_H_
#define SHERPA_NCNN_CSRC_VOICE_ACTIVITY_DETECTOR_H_

#include <memory>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-ncnn/csrc/silero-vad-model-config.h"

namespace sherpa_ncnn {

struct SpeechSegment {
  int32_t start;  // in samples
  std::vector<float> samples;
};

class VoiceActivityDetector {
 public:
  explicit VoiceActivityDetector(const SileroVadModelConfig &config,
                                 float buffer_size_in_seconds = 60);

#if __ANDROID_API__ >= 9
  VoiceActivityDetector(AAssetManager *mgr, const SileroVadModelConfig &config,
                        float buffer_size_in_seconds = 60);
#endif

  ~VoiceActivityDetector();

  void AcceptWaveform(const float *samples, int32_t n);
  bool Empty() const;
  void Pop();
  void Clear();
  const SpeechSegment &Front() const;

  bool IsSpeechDetected() const;

  void Reset() const;

  // At the end of the utterance, you can invoke this method so that
  // the last speech segment can be detected.
  void Flush() const;

  const SileroVadModelConfig &GetConfig() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_VOICE_ACTIVITY_DETECTOR_H_
