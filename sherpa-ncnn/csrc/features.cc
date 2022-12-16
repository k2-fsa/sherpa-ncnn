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

#include "sherpa-ncnn/csrc/features.h"

#include <algorithm>
#include <memory>

#include "mat.h"  // NOLINT

namespace sherpa_ncnn {

FeatureExtractor::FeatureExtractor(const knf::FbankOptions &opts)
    : opts_(opts) {
  fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
}

void FeatureExtractor::AcceptWaveform(float sampling_rate,
                                      const float *waveform, int32_t n) {
  std::lock_guard<std::mutex> lock(mutex_);
  fbank_->AcceptWaveform(sampling_rate, waveform, n);
}

void FeatureExtractor::InputFinished() {
  std::lock_guard<std::mutex> lock(mutex_);
  fbank_->InputFinished();
}

int32_t FeatureExtractor::NumFramesReady() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return fbank_->NumFramesReady();
}

bool FeatureExtractor::IsLastFrame(int32_t frame) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return fbank_->IsLastFrame(frame);
}

ncnn::Mat FeatureExtractor::GetFrames(int32_t frame_index, int32_t n) const {
  if (frame_index + n > NumFramesReady()) {
    fprintf(stderr, "%d + %d > %d\n", frame_index, n, NumFramesReady());
    exit(-1);
  }
  std::lock_guard<std::mutex> lock(mutex_);

  int32_t feature_dim = fbank_->Dim();
  ncnn::Mat features;
  features.create(feature_dim, n);

  for (int32_t i = 0; i != n; ++i) {
    const float *f = fbank_->GetFrame(i + frame_index);
    std::copy(f, f + feature_dim, features.row(i));
  }

  return features;
}

void FeatureExtractor::Reset() {
  fbank_ = std::make_unique<knf::OnlineFbank>(opts_);
}

}  // namespace sherpa_ncnn
