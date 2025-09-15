// sherpa-ncnn/csrc/offline-recognizer.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-recognizer.h"

#include <memory>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-ncnn/csrc/file-utils.h"
#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/offline-recognizer-impl.h"
#include "sherpa-ncnn/csrc/text-utils.h"

namespace sherpa_ncnn {

void OfflineRecognizerConfig::Register(ParseOptions *po) {
  feat_config.Register(po);
  model_config.Register(po);

  po->Register("decoding-method", &decoding_method,
               "decoding method,"
               "Valid values: greedy_search.");

  po->Register("blank-penalty", &blank_penalty,
               "The penalty applied on blank symbol during decoding. "
               "Note: It is a positive value. "
               "Increasing value will lead to lower deletion at the cost"
               "of higher insertions. "
               "Currently only applicable for transducer models.");
}

bool OfflineRecognizerConfig::Validate() const {
  return model_config.Validate();
}

std::string OfflineRecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineRecognizerConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "decoding_method=\"" << decoding_method << "\", ";
  os << "blank_penalty=" << blank_penalty << ")";

  return os.str();
}

template <typename Manager>
OfflineRecognizer::OfflineRecognizer(Manager *mgr,
                                     const OfflineRecognizerConfig &config)
    : impl_(OfflineRecognizerImpl::Create(mgr, config)) {}

OfflineRecognizer::OfflineRecognizer(const OfflineRecognizerConfig &config)
    : impl_(OfflineRecognizerImpl::Create(config)) {}

OfflineRecognizer::~OfflineRecognizer() = default;

std::unique_ptr<OfflineStream> OfflineRecognizer::CreateStream() const {
  return impl_->CreateStream();
}

void OfflineRecognizer::DecodeStreams(OfflineStream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

void OfflineRecognizer::SetConfig(const OfflineRecognizerConfig &config) {
  impl_->SetConfig(config);
}

OfflineRecognizerConfig OfflineRecognizer::GetConfig() const {
  return impl_->GetConfig();
}

#if __ANDROID_API__ >= 9
template OfflineRecognizer::OfflineRecognizer(
    AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

#if __OHOS__
template OfflineRecognizer::OfflineRecognizer(
    NativeResourceManager *mgr, const OfflineRecognizerConfig &config);
#endif

}  // namespace sherpa_ncnn
