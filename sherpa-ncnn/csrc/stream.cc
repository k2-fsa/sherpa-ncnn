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

#include "sherpa-ncnn/csrc/stream.h"

namespace sherpa_ncnn {

class Stream::Impl {
 public:
  explicit Impl(const FeatureExtractorConfig &config,
                ContextGraphPtr context_graph)
      : feat_extractor_(config), context_graph_(context_graph) {}

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    feat_extractor_.AcceptWaveform(sampling_rate, waveform, n);
  }

  void InputFinished() { feat_extractor_.InputFinished(); }

  int32_t NumFramesReady() const {
    return feat_extractor_.NumFramesReady() - start_frame_index_;
  }

  bool IsLastFrame(int32_t frame) const {
    return feat_extractor_.IsLastFrame(frame);
  }

  ncnn::Mat GetFrames(int32_t frame_index, int32_t n) const {
    return feat_extractor_.GetFrames(frame_index + start_frame_index_, n);
  }

  void Reset() {
    start_frame_index_ += num_processed_frames_;
    num_processed_frames_ = 0;
  }

  int32_t &GetNumProcessedFrames() { return num_processed_frames_; }

  void SetResult(const DecoderResult &r) {
    int32_t offset = result_.frame_offset;
    result_ = r;
    result_.frame_offset = offset;
  }

  DecoderResult &GetResult() { return result_; }

  void SetStates(const std::vector<ncnn::Mat> &states) { states_ = states; }

  std::vector<ncnn::Mat> &GetStates() { return states_; }

  const ContextGraphPtr &GetContextGraph() const { return context_graph_; }

 private:
  FeatureExtractor feat_extractor_;
  ContextGraphPtr context_graph_;
  int32_t num_processed_frames_ = 0;  // before subsampling
  int32_t start_frame_index_ = 0;
  DecoderResult result_;
  std::vector<ncnn::Mat> states_;
};

Stream::Stream(const FeatureExtractorConfig &config,
               ContextGraphPtr context_graph)
    : impl_(std::make_unique<Impl>(config, context_graph)) {}

Stream::~Stream() = default;

void Stream::AcceptWaveform(int32_t sampling_rate, const float *waveform,
                            int32_t n) {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

void Stream::InputFinished() { impl_->InputFinished(); }

int32_t Stream::NumFramesReady() const { return impl_->NumFramesReady(); }

bool Stream::IsLastFrame(int32_t frame) const {
  return impl_->IsLastFrame(frame);
}

ncnn::Mat Stream::GetFrames(int32_t frame_index, int32_t n) const {
  return impl_->GetFrames(frame_index, n);
}

void Stream::Reset() { impl_->Reset(); }

int32_t &Stream::GetNumProcessedFrames() {
  return impl_->GetNumProcessedFrames();
}

void Stream::SetResult(const DecoderResult &r) { impl_->SetResult(r); }

DecoderResult &Stream::GetResult() { return impl_->GetResult(); }

void Stream::SetStates(const std::vector<ncnn::Mat> &states) {
  impl_->SetStates(states);
}

std::vector<ncnn::Mat> &Stream::GetStates() { return impl_->GetStates(); }

const ContextGraphPtr &Stream::GetContextGraph() const {
  return impl_->GetContextGraph();
}
}  // namespace sherpa_ncnn
