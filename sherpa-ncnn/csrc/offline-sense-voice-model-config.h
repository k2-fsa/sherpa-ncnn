// sherpa-ncnn/csrc/offline-sense-voice-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_NCNN_CSRC_OFFLINE_SENSE_VOICE_MODEL_CONFIG_H_
#define SHERPA_NCNN_CSRC_OFFLINE_SENSE_VOICE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-ncnn/csrc/parse-options.h"

namespace sherpa_ncnn {

struct OfflineSenseVoiceModelConfig {
  // it should contain model.ncnn.param and model.ncnn.bin
  std::string model_dir;

  // "" or "auto" to let the model recognize the language
  // valid values:
  //  zh, en, ja, ko, yue, auto
  std::string language = "auto";

  // true to use inverse text normalization
  // false to not use inverse text normalization
  bool use_itn = false;

  OfflineSenseVoiceModelConfig() = default;
  OfflineSenseVoiceModelConfig(const std::string &model_dir,
                               const std::string &language, bool use_itn)
      : model_dir(model_dir), language(language), use_itn(use_itn) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_SENSE_VOICE_MODEL_CONFIG_H_
