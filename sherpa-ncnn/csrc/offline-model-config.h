// sherpa-ncnn/csrc/offline-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_NCNN_CSRC_OFFLINE_MODEL_CONFIG_H_
#define SHERPA_NCNN_CSRC_OFFLINE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-ncnn/csrc/offline-sense-voice-model-config.h"

namespace sherpa_ncnn {

struct OfflineModelConfig {
  OfflineSenseVoiceModelConfig sense_voice;

  std::string tokens;
  int32_t num_threads = 2;
  bool debug = false;

  OfflineModelConfig() = default;
  OfflineModelConfig(const OfflineSenseVoiceModelConfig &sense_voice,
                     const std::string &tokens, int32_t num_threads, bool debug)
      : sense_voice(sense_voice),
        tokens(tokens),
        num_threads(num_threads),
        debug(debug) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_MODEL_CONFIG_H_
