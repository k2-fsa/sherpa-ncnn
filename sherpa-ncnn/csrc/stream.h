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

#ifndef SHERPA_NCNN_CSRC_STREAM_H_
#define SHERPA_NCNN_CSRC_STREAM_H_

#include <memory>
#include <vector>

#include "sherpa-ncnn/csrc/context-graph.h"
#include "sherpa-ncnn/csrc/decoder.h"
#include "sherpa-ncnn/csrc/features.h"

namespace sherpa_ncnn {
class Stream {
 public:
  explicit Stream(const FeatureExtractorConfig &config = {},
                  ContextGraphPtr context_graph = nullptr);
  ~Stream();

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

  /**
   * InputFinished() tells the class you won't be providing any
   * more waveform.  This will help flush out the last frame or two
   * of features, in the case where snip-edges == false; it also
   * affects the return value of IsLastFrame().
   */
  void InputFinished();

  int32_t NumFramesReady() const;

  /** Note: IsLastFrame() will only ever return true if you have called
   * InputFinished() (and this frame is the last frame).
   */
  bool IsLastFrame(int32_t frame) const;

  /** Get n frames starting from the given frame index.
   *
   * @param frame_index  The starting frame index
   * @param n  Number of frames to get.
   * @return Return a 2-D tensor of shape (n, feature_dim).
   *         which is flattened into a 1-D vector (flattened in in row major)
   */
  ncnn::Mat GetFrames(int32_t frame_index, int32_t n) const;

  void Reset();

  /**
   * Finalize the decoding result. This is mainly for decoding with hotwords
   * (i.e. providing context_graph). It will cancel the boosting score of the
   * partial matching paths. For example, the hotword is "BCD", the path "ABC"
   * gets boosting score of "BC" but it fails to match the whole hotword "BCD",
   * so we have to cancel the scores of "BC" at the end.
   */
  void Finalize();

  // Return a reference to the number of processed frames so far
  // before subsampling..
  // Initially, it is 0. It is always less than NumFramesReady().
  //
  // The returned reference is valid as long as this object is alive.
  int32_t &GetNumProcessedFrames();

  void SetResult(const DecoderResult &r);
  DecoderResult &GetResult();

  void SetStates(const std::vector<ncnn::Mat> &states);
  std::vector<ncnn::Mat> &GetStates();
  /**
   * Get the context graph corresponding to this stream.
   *
   * @return Return the context graph for this stream.
   */
  const ContextGraphPtr &GetContextGraph() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_STREAM_H_
