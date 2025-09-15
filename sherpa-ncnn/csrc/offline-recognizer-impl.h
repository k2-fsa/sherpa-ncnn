// sherpa-ncnn/csrc/offline-recognizer-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_OFFLINE_RECOGNIZER_IMPL_H_
#define SHERPA_NCNN_CSRC_OFFLINE_RECOGNIZER_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/offline-recognizer.h"
#include "sherpa-ncnn/csrc/offline-stream.h"

namespace sherpa_ncnn {

class OfflineRecognizerImpl {
 public:
  static std::unique_ptr<OfflineRecognizerImpl> Create(
      const OfflineRecognizerConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineRecognizerImpl> Create(
      Manager *mgr, const OfflineRecognizerConfig &config);

  virtual ~OfflineRecognizerImpl() = default;

  virtual std::unique_ptr<OfflineStream> CreateStream() const = 0;

  virtual void DecodeStreams(OfflineStream **ss, int32_t n) const = 0;

  virtual void SetConfig(const OfflineRecognizerConfig &config) = 0;
  virtual OfflineRecognizerConfig GetConfig() const = 0;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_RECOGNIZER_IMPL_H_
