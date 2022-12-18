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

#ifndef SHERPA_NCNN_CSRC_FEATURES_H_
#define SHERPA_NCNN_CSRC_FEATURES_H_

#include <memory>
#include <mutex>  // NOLINT

#include "kaldi-native-fbank/csrc/online-feature.h"

namespace ncnn {
class Mat;
}

namespace sherpa_ncnn {

class FeatureExtractor {
 public:
  explicit FeatureExtractor(const knf::FbankOptions &fbank_opts);

  /**
     @param sampling_rate The sampling_rate of the input waveform. Should match
                          the one expected by the feature extractor.
     @param waveform Pointer to a 1-D array of size n
     @param n Number of entries in waveform
   */
  void AcceptWaveform(float sampling_rate, const float *waveform, int32_t n);

  // InputFinished() tells the class you won't be providing any
  // more waveform.  This will help flush out the last frame or two
  // of features, in the case where snip-edges == false; it also
  // affects the return value of IsLastFrame().
  void InputFinished();

  int32_t NumFramesReady() const;

  // Note: IsLastFrame() will only ever return true if you have called
  // InputFinished() (and this frame is the last frame).
  bool IsLastFrame(int32_t frame) const;

  /** Get n frames starting from the given frame index.
   *
   * @param frame_index  The starting frame index
   * @param n  Number of frames to get.
   * @return Return a 2-D tensor of shape (n, feature_dim).
   *         ans.w == feature_dim; ans.h == n
   */
  ncnn::Mat GetFrames(int32_t frame_index, int32_t n) const;

  void Reset();

 private:
  std::unique_ptr<knf::OnlineFbank> fbank_;
  knf::FbankOptions opts_;
  mutable std::mutex mutex_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_FEATURES_H_
