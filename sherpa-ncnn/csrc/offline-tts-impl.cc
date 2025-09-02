// sherpa-ncnn/csrc/offline-tts-impl.cc
//
// Copyright (c)  2023-2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-tts-impl.h"

#include <memory>
#include <vector>

#include "sherpa-ncnn/csrc/offline-tts-vits-impl.h"

namespace sherpa_ncnn {

std::unique_ptr<OfflineTtsImpl> OfflineTtsImpl::Create(
    const OfflineTtsConfig &config) {
  if (!config.model.vits.model_dir.empty()) {
    return std::make_unique<OfflineTtsVitsImpl>(config);
  }

  SHERPA_NCNN_LOGE("Please provide a tts model.");

  return {};
}

}  // namespace sherpa_ncnn
