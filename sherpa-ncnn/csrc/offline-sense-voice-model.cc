// sherpa-ncnn/csrc/offline-sense-voice-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-sense-voice-model.h"

#include <math.h>

#include <algorithm>
#include <mutex>  // NOLINT
#include <string>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "mat.h"  // NOLINT
#include "net.h"  // NOLINT
#include "sherpa-ncnn/csrc/file-utils.h"
#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/text-utils.h"

namespace sherpa_ncnn {

namespace {

class SinusoidalPositionEncoder {
 public:
  explicit SinusoidalPositionEncoder(int32_t dim) : dim_(dim) {}

  ncnn::Mat operator()(int32_t len) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (pos_.h < len) {
      Resize(len);
    }

    return pos_.row_range(0, len).clone();
  }

  void Resize(int32_t len) {
    int32_t timesteps = len;
    int32_t input_dim = dim_;

    pos_.create(input_dim, timesteps);

    int32_t half_dim = input_dim / 2;
    float log_timescale_increment = logf(10000.f) / (half_dim - 1);

    float *outptr = pos_;

    for (int32_t t = 0; t < timesteps; ++t) {
      int32_t pos = t + 1;  // positions start from 1

      for (int32_t i = 0; i < half_dim; i++) {
        float inv_timescale = expf(-i * log_timescale_increment);

        float scaled_time = pos * inv_timescale;

        float sinv = sinf(scaled_time);
        float cosv = cosf(scaled_time);

        // write both sin and cos channels
        outptr[t * input_dim + i] = sinv;
        outptr[t * input_dim + i + half_dim] = cosv;
      }
    }
  }

 private:
  std::mutex mutex_;
  int32_t max_len_;
  int32_t dim_;
  ncnn::Mat pos_;
};

}  // namespace

class OfflineSenseVoiceModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config), pos_encoder_(560) {
    Init();
  }

  ncnn::Mat Forward(ncnn::Mat features, int32_t language, int32_t text_norm) {
    ncnn::Mat prompt(4);
    int32_t *p_prompt = prompt;
    p_prompt[0] = language;
    p_prompt[1] = 1;
    p_prompt[2] = 2;
    p_prompt[3] = text_norm;

    ncnn::Mat pos = pos_encoder_(features.h + 4);

    ncnn::Extractor ex = net_.create_extractor();

    ex.input("in0", features);
    ex.input("in1", prompt);
    ex.input("in2", pos);

    ncnn::Mat logits;

    ex.extract("out0", logits);

    return logits;
  }

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

 private:
  void Init() {
    net_.opt.num_threads = config_.num_threads;

    std::string param = config_.sense_voice.model_dir + "/model.ncnn.param";
    std::string bin = config_.sense_voice.model_dir + "/model.ncnn.bin";

    net_.load_param(param.c_str());
    net_.load_model(bin.c_str());

    meta_data_.vocab_size = 25055;
    meta_data_.window_size = 7;
    meta_data_.window_shift = 6;
    meta_data_.normalize_samples = 0;
    meta_data_.with_itn_id = 14;
    meta_data_.without_itn_id = 15;

    int32_t lang_auto = 0;
    int32_t lang_zh = 3;
    int32_t lang_en = 4;
    int32_t lang_ja = 11;
    int32_t lang_ko = 12;
    int32_t lang_yue = 7;

    meta_data_.lang2id = {
        {"auto", lang_auto}, {"zh", lang_zh}, {"en", lang_en},
        {"ja", lang_ja},     {"ko", lang_ko}, {"yue", lang_yue},
    };
  }

 private:
  OfflineModelConfig config_;
  SinusoidalPositionEncoder pos_encoder_;

  ncnn::Net net_;

  OfflineSenseVoiceModelMetaData meta_data_;
};

OfflineSenseVoiceModel::OfflineSenseVoiceModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSenseVoiceModel::OfflineSenseVoiceModel(Manager *mgr,
                                               const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineSenseVoiceModel::~OfflineSenseVoiceModel() = default;

ncnn::Mat OfflineSenseVoiceModel::Forward(const ncnn::Mat &features,
                                          int32_t language,
                                          int32_t text_norm) const {
  return impl_->Forward(features, language, text_norm);
}

const OfflineSenseVoiceModelMetaData &OfflineSenseVoiceModel::GetModelMetadata()
    const {
  return impl_->GetModelMetadata();
}

#if __ANDROID_API__ >= 9
template OfflineSenseVoiceModel::OfflineSenseVoiceModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineSenseVoiceModel::OfflineSenseVoiceModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_ncnn
