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

#include "sherpa-ncnn/csrc/silero-vad-model.h"

#include <vector>

#include "net.h"  // NOLINT
#include "sherpa-ncnn/csrc/model.h"
#include "sherpa-ncnn/csrc/silero-vad-model-config.h"

namespace sherpa_ncnn {

class SileroVadModel::Impl {
 public:
  explicit Impl(const SileroVadModelConfig &config) : config_(config) {
    model_.opt = config.opt;
    bool has_gpu = false;

#if NCNN_VULKAN
    has_gpu = ncnn::get_gpu_count() > 0;
#endif

    if (has_gpu && config_.use_vulkan_compute) {
      model_.opt.use_vulkan_compute = true;
      NCNN_LOGE("Use GPU");
    }

    Model::InitNet(model_, config_.param, config_.bin);
    PostInit();
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const SileroVadModelConfig &config)
      : config_(config), {
    model_.opt = config.opt;
    bool has_gpu = false;

#if NCNN_VULKAN
    has_gpu = ncnn::get_gpu_count() > 0;
#endif

    if (has_gpu && config_.use_vulkan_compute) {
      model_.opt.use_vulkan_compute = true;
      NCNN_LOGE("Use GPU");
    }

    Model::InitNet(mgr, model_, config_.param, config_.bin);

    PostInit();
  }
#endif

  void Reset() {
    ResetV4();

    triggered_ = false;
    current_sample_ = 0;
    temp_start_ = 0;
    temp_end_ = 0;
  }

  bool IsSpeech(const float *samples, int32_t n) {
    if (n != WindowSize()) {
      NCNN_LOGE("n: %d != window_size: %d", n, WindowSize());
      exit(-1);
    }

    float prob = Run(samples, n);

    float threshold = config_.threshold;

    current_sample_ += config_.window_size;

    if (prob > threshold && temp_end_ != 0) {
      temp_end_ = 0;
    }

    if (prob > threshold && temp_start_ == 0) {
      // start speaking, but we require that it must satisfy
      // min_speech_duration
      temp_start_ = current_sample_;
      return false;
    }

    if (prob > threshold && temp_start_ != 0 && !triggered_) {
      if (current_sample_ - temp_start_ < min_speech_samples_) {
        return false;
      }

      triggered_ = true;

      return true;
    }

    if ((prob < threshold) && !triggered_) {
      // silence
      temp_start_ = 0;
      temp_end_ = 0;
      return false;
    }

    if ((prob > threshold - 0.15) && triggered_) {
      // speaking
      return true;
    }

    if ((prob > threshold) && !triggered_) {
      // start speaking
      triggered_ = true;

      return true;
    }

    if ((prob < threshold) && triggered_) {
      // stop to speak
      if (temp_end_ == 0) {
        temp_end_ = current_sample_;
      }

      if (current_sample_ - temp_end_ < min_silence_samples_) {
        // continue speaking
        return true;
      }
      // stopped speaking
      temp_start_ = 0;
      temp_end_ = 0;
      triggered_ = false;
      return false;
    }

    return false;
  }

  int32_t WindowShift() const { return config_.window_size; }

  int32_t WindowSize() const { return config_.window_size; }

  int32_t MinSilenceDurationSamples() const { return min_silence_samples_; }

  int32_t MinSpeechDurationSamples() const { return min_speech_samples_; }

  void SetMinSilenceDuration(float s) {
    min_silence_samples_ = config_.sample_rate * s;
  }

  void SetThreshold(float threshold) { config_.threshold = threshold; }

 private:
  void PostInit() {
    min_silence_samples_ = config_.sample_rate * config_.min_silence_duration;

    min_speech_samples_ = config_.sample_rate * config_.min_speech_duration;

    // input indexes map
    // [0] -> in0, x
    // [1] -> in1, h
    // [2] -> in2, c
    input_indexes_.resize(4);

    // output indexes map
    // [0] -> out0, prob
    // [1] -> out1, h
    // [2] -> out2, c
    output_indexes_.resize(3);

    const auto &blobs = model_.blobs();
    for (int32_t i = 0; i != blobs.size(); ++i) {
      const auto &b = blobs[i];
      if (b.name == "in0") input_indexes_[0] = i;
      if (b.name == "in1") input_indexes_[1] = i;
      if (b.name == "in2") input_indexes_[2] = i;
      if (b.name == "out0") output_indexes_[0] = i;
      if (b.name == "out1") output_indexes_[1] = i;
      if (b.name == "out2") output_indexes_[2] = i;
    }

    h_ = ncnn::Mat(64, 1, 2);
    c_ = ncnn::Mat(64, 1, 2);

    h_.fill(0);
    c_.fill(0);
  }

  void ResetV4() {
    h_.fill(0);
    c_.fill(0);
  }

  float Run(const float *samples, int32_t n) {
    // TODO(fangjun): Support V5
    return RunV4(samples, n);
  }

  float RunV4(const float *samples, int32_t n) {
    ncnn::Mat x(n, 1, 1, const_cast<float *>(samples));

    ncnn::Extractor ex = model_.create_extractor();

    ex.input(input_indexes_[0], x);
    ex.input(input_indexes_[1], h_);
    ex.input(input_indexes_[2], c_);

    ncnn::Mat out;
    ex.extract(output_indexes_[0], out);
    ex.extract(output_indexes_[1], h_);
    ex.extract(output_indexes_[2], c_);

    float prob = out[0];
    return prob;
  }

 private:
  ncnn::Net model_;
  std::vector<int32_t> input_indexes_;
  std::vector<int32_t> output_indexes_;

  ncnn::Mat h_;
  ncnn::Mat c_;

  SileroVadModelConfig config_;

  int32_t min_silence_samples_;
  int32_t min_speech_samples_;

  bool triggered_ = false;
  int32_t current_sample_ = 0;
  int32_t temp_start_ = 0;
  int32_t temp_end_ = 0;
};

SileroVadModel::SileroVadModel(const SileroVadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
SileroVadModel::SileroVadModel(AAssetManager *mgr,
                               const SileroVadModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

SileroVadModel::~SileroVadModel() = default;

void SileroVadModel::Reset() { return impl_->Reset(); }

bool SileroVadModel::IsSpeech(const float *samples, int32_t n) {
  return impl_->IsSpeech(samples, n);
}

int32_t SileroVadModel::WindowSize() const { return impl_->WindowSize(); }

int32_t SileroVadModel::WindowShift() const { return impl_->WindowShift(); }

int32_t SileroVadModel::MinSilenceDurationSamples() const {
  return impl_->MinSilenceDurationSamples();
}

int32_t SileroVadModel::MinSpeechDurationSamples() const {
  return impl_->MinSpeechDurationSamples();
}

void SileroVadModel::SetMinSilenceDuration(float s) {
  impl_->SetMinSilenceDuration(s);
}

void SileroVadModel::SetThreshold(float threshold) {
  impl_->SetThreshold(threshold);
}

}  // namespace sherpa_ncnn
