// sherpa-ncnn/csrc/offline-recognizer-impl.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-recognizer-impl.h"

#include <string>
#include <strstream>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "sherpa-ncnn/csrc/file-utils.h"
#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/offline-recognizer-sense-voice-impl.h"
#include "sherpa-ncnn/csrc/text-utils.h"

namespace sherpa_ncnn {

std::unique_ptr<OfflineRecognizerImpl> OfflineRecognizerImpl::Create(
    const OfflineRecognizerConfig &config) {
  if (!config.model_config.sense_voice.model_dir.empty()) {
    return std::make_unique<OfflineRecognizerSenseVoiceImpl>(config);
  }

  SHERPA_NCNN_LOGE("Please provide a model!");

  return nullptr;
}

template <typename Manager>
std::unique_ptr<OfflineRecognizerImpl> OfflineRecognizerImpl::Create(
    Manager *mgr, const OfflineRecognizerConfig &config) {
  if (!config.model_config.sense_voice.model_dir.empty()) {
    return std::make_unique<OfflineRecognizerSenseVoiceImpl>(mgr, config);
  }

  SHERPA_NCNN_LOGE("Please provide a model!");

  return nullptr;
}

#if __ANDROID_API__ >= 9
template OfflineRecognizerImpl::OfflineRecognizerImpl(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);

template std::unique_ptr<OfflineRecognizerImpl> OfflineRecognizerImpl::Create(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

}  // namespace sherpa_ncnn
