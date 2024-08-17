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
#include "sherpa-ncnn/csrc/voice-activity-detector.h"

#include <algorithm>
#include <queue>
#include <utility>

#include "sherpa-ncnn/csrc/circular-buffer.h"
#include "sherpa-ncnn/csrc/silero-vad-model.h"

namespace sherpa_ncnn {

class VoiceActivityDetector::Impl {
 public:
  explicit Impl(const SileroVadModelConfig &config,
                float buffer_size_in_seconds = 60)
      : model_(std::make_unique<SileroVadModel>(config)),
        config_(config),
        buffer_(buffer_size_in_seconds * config.sample_rate) {}

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const SileroVadModelConfig &config,
       float buffer_size_in_seconds = 60)
      : model_(std::make_unique<SileroVadModel>(mgr, config)),
        config_(config),
        buffer_(buffer_size_in_seconds * config.sample_rate) {}
#endif

  void AcceptWaveform(const float *samples, int32_t n) {
    if (buffer_.Size() > max_utterance_length_) {
      model_->SetMinSilenceDuration(new_min_silence_duration_s_);
      model_->SetThreshold(new_threshold_);
    } else {
      model_->SetMinSilenceDuration(config_.min_silence_duration);
      model_->SetThreshold(config_.threshold);
    }

    int32_t window_size = model_->WindowSize();
    int32_t window_shift = model_->WindowShift();

    // note n is usually window_size and there is no need to use
    // an extra buffer here
    last_.insert(last_.end(), samples, samples + n);

    if (last_.size() < window_size) {
      return;
    }

    // Note: For v4, window_shift == window_size
    int32_t k =
        (static_cast<int32_t>(last_.size()) - window_size) / window_shift + 1;
    const float *p = last_.data();
    bool is_speech = false;

    for (int32_t i = 0; i < k; ++i, p += window_shift) {
      buffer_.Push(p, window_shift);
      // NOTE(fangjun): Please don't use a very large n.
      bool this_window_is_speech = model_->IsSpeech(p, window_size);
      is_speech = is_speech || this_window_is_speech;
    }

    last_ = std::vector<float>(
        p, static_cast<const float *>(last_.data()) + last_.size());

    if (is_speech) {
      if (start_ == -1) {
        // beginning of speech
        start_ = std::max(buffer_.Tail() - 2 * model_->WindowSize() -
                              model_->MinSpeechDurationSamples(),
                          buffer_.Head());
      }
    } else {
      // non-speech
      if (start_ != -1 && buffer_.Size()) {
        // end of speech, save the speech segment
        int32_t end = buffer_.Tail() - model_->MinSilenceDurationSamples();

        std::vector<float> s = buffer_.Get(start_, end - start_);
        SpeechSegment segment;

        segment.start = start_;
        segment.samples = std::move(s);

        segments_.push(std::move(segment));

        buffer_.Pop(end - buffer_.Head());
      }

      if (start_ == -1) {
        int32_t end = buffer_.Tail() - 2 * model_->WindowSize() -
                      model_->MinSpeechDurationSamples();
        int32_t n = std::max(0, end - buffer_.Head());
        if (n > 0) {
          buffer_.Pop(n);
        }
      }

      start_ = -1;
    }
  }

  bool Empty() const { return segments_.empty(); }

  void Pop() { segments_.pop(); }

  void Clear() { std::queue<SpeechSegment>().swap(segments_); }

  const SpeechSegment &Front() const { return segments_.front(); }

  void Reset() {
    std::queue<SpeechSegment>().swap(segments_);

    model_->Reset();
    buffer_.Reset();

    start_ = -1;
  }

  void Flush() {
    if (start_ == -1 || buffer_.Size() == 0) {
      return;
    }

    int32_t end = buffer_.Tail() - model_->MinSilenceDurationSamples();
    if (end <= start_) {
      return;
    }

    std::vector<float> s = buffer_.Get(start_, end - start_);

    SpeechSegment segment;

    segment.start = start_;
    segment.samples = std::move(s);

    segments_.push(std::move(segment));

    buffer_.Pop(end - buffer_.Head());
    start_ = -1;
  }

  bool IsSpeechDetected() const { return start_ != -1; }

  const SileroVadModelConfig &GetConfig() const { return config_; }

 private:
  std::queue<SpeechSegment> segments_;

  std::unique_ptr<SileroVadModel> model_;
  SileroVadModelConfig config_;
  CircularBuffer buffer_;
  std::vector<float> last_;

  int max_utterance_length_ = 16000 * 20;  // in samples
  float new_min_silence_duration_s_ = 0.1;
  float new_threshold_ = 1.10;

  int32_t start_ = -1;
};

VoiceActivityDetector::VoiceActivityDetector(
    const SileroVadModelConfig &config, float buffer_size_in_seconds /*= 60*/)
    : impl_(std::make_unique<Impl>(config, buffer_size_in_seconds)) {}

#if __ANDROID_API__ >= 9
VoiceActivityDetector::VoiceActivityDetector(
    AAssetManager *mgr, const SileroVadModelConfig &config,
    float buffer_size_in_seconds /*= 60*/)
    : impl_(std::make_unique<Impl>(mgr, config, buffer_size_in_seconds)) {}
#endif

VoiceActivityDetector::~VoiceActivityDetector() = default;

void VoiceActivityDetector::AcceptWaveform(const float *samples, int32_t n) {
  impl_->AcceptWaveform(samples, n);
}

bool VoiceActivityDetector::Empty() const { return impl_->Empty(); }

void VoiceActivityDetector::Pop() { impl_->Pop(); }

void VoiceActivityDetector::Clear() { impl_->Clear(); }

const SpeechSegment &VoiceActivityDetector::Front() const {
  return impl_->Front();
}

void VoiceActivityDetector::Reset() const { impl_->Reset(); }

void VoiceActivityDetector::Flush() const { impl_->Flush(); }

bool VoiceActivityDetector::IsSpeechDetected() const {
  return impl_->IsSpeechDetected();
}

const SileroVadModelConfig &VoiceActivityDetector::GetConfig() const {
  return impl_->GetConfig();
}

}  // namespace sherpa_ncnn
