// sherpa-ncnn/csrc/offline-tts-vits-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_MODEL_CONFIG_H_
#define SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_MODEL_CONFIG_H_

#include <string>

#include "sherpa-ncnn/csrc/parse-options.h"

namespace sherpa_ncnn {

struct OfflineTtsVitsModelConfig {
  // We assume there are at least the following files inside
  // the model dir:
  //  - config.json
  //  - lexicon.txt
  //  - encoder.ncnn.{param,bin}
  //  - dp.ncnn.{param,bin}
  //  - flow.ncnn.{param,bin}
  //  - decoder.ncnn.{param,bin}
  std::string model_dir;

  OfflineTtsVitsModelConfig() = default;

  explicit OfflineTtsVitsModelConfig(const std::string &model_dir)
      : model_dir(model_dir) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_MODEL_CONFIG_H_
