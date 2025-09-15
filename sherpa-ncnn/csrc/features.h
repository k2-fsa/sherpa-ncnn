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
#include <string>

#include "sherpa-ncnn/csrc/parse-options.h"

namespace ncnn {
class Mat;
}

namespace sherpa_ncnn {

struct FeatureExtractorConfig {
  int32_t sampling_rate = 16000;
  int32_t feature_dim = 80;

  // minimal frequency for Mel-filterbank, in Hz
  float low_freq = 20.0f;

  // maximal frequency of Mel-filterbank
  // in Hz; negative value is subtracted from Nyquist freq.:
  // i.e. for sampling_rate 16000 / 2 - 400 = 7600Hz
  //
  // Please see
  // https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/fbank.py#L27
  // and
  // https://github.com/k2-fsa/sherpa-onnx/issues/514
  float high_freq = -400.0f;

  // dithering constant, useful for signals with hard-zeroes in non-speech parts
  // this prevents large negative values in log-mel filterbanks
  //
  // In k2, audio samples are in range [-1..+1], in kaldi the range was
  // [-32k..+32k], so the value 0.00003 is equivalent to kaldi default 1.0
  //
  float dither = 0.0f;  // dithering disabled by default

  // Set internally by some models, e.g., paraformer sets it to false.
  // This parameter is not exposed to users from the commandline
  // If true, the feature extractor expects inputs to be normalized to
  // the range [-1, 1].
  // If false, we will multiply the inputs by 32768
  bool normalize_samples = true;

  bool snip_edges = false;
  float frame_shift_ms = 10.0f;   // in milliseconds.
  float frame_length_ms = 25.0f;  // in milliseconds.
  bool is_librosa = false;
  bool remove_dc_offset = true;       // Subtract mean of wave before FFT.
  float preemph_coeff = 0.97f;        // Preemphasis coefficient.
  std::string window_type = "povey";  // e.g. Hamming window
  bool round_to_power_of_two = true;

  std::string ToString() const;

  void Register(ParseOptions *po);
};

class FeatureExtractor {
 public:
  explicit FeatureExtractor(const FeatureExtractorConfig &config);
  ~FeatureExtractor();

  /**
     @param sampling_rate The sampling_rate of the input waveform. We will
                          do resample if it is different from
                          config.sampling_rate.
                          Caution: You MUST not use a different sampling rate
                          across different calls for AcceptWaveform().
     @param waveform Pointer to a 1-D array of size n
     @param n Number of entries in waveform
   */
  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n);

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

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_FEATURES_H_
