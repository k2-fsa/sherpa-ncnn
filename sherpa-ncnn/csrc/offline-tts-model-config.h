// sherpa-ncnn/csrc/offline-tts-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_OFFLINE_TTS_MODEL_CONFIG_H_
#define SHERPA_NCNN_CSRC_OFFLINE_TTS_MODEL_CONFIG_H_

#include <string>

#include "sherpa-ncnn/csrc/offline-tts-vits-model-config.h"
#include "sherpa-ncnn/csrc/parse-options.h"

namespace sherpa_ncnn {

struct OfflineTtsModelConfig {
  OfflineTtsVitsModelConfig vits;

  int32_t num_threads = 1;
  bool debug = false;

  OfflineTtsModelConfig() = default;

  OfflineTtsModelConfig(const OfflineTtsVitsModelConfig &vits,
                        int32_t num_threads, bool debug)
      : vits(vits), num_threads(num_threads), debug(debug) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_TTS_MODEL_CONFIG_H_
