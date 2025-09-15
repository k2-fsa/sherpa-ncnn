// sherpa-ncnn/csrc/offline-sense-voice-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-sense-voice-model-config.h"

#include "sherpa-ncnn/csrc/file-utils.h"
#include "sherpa-ncnn/csrc/macros.h"

namespace sherpa_ncnn {

void OfflineSenseVoiceModelConfig::Register(ParseOptions *po) {
  po->Register("sense-voice-model-dir", &model_dir,
               "Path to SenseVoice model directory. It should contain "
               "model.ncnn.param and model.ncnn.bin.");

  po->Register(
      "sense-voice-language", &language,
      "Valid values: auto, zh, en, ja, ko, yue. If left empty, auto is used");

  po->Register(
      "sense-voice-use-itn", &use_itn,
      "True to enable inverse text normalization. False to disable it.");
}

bool OfflineSenseVoiceModelConfig::Validate() const {
  if (model_dir.empty()) {
    SHERPA_NCNN_LOGE("Please provide --sense-voice-model_dir");
    return false;
  }

  std::vector<std::string> files_to_check = {
      "model.ncnn.param",
      "model.ncnn.bin",
  };

  bool ok = true;
  for (const auto &f : files_to_check) {
    auto name = model_dir + "/" + f;
    if (!FileExists(name)) {
      SHERPA_NCNN_LOGE("'%s' does not exist inside the directory '%s'",
                       name.c_str(), model_dir.c_str());
      ok = false;
    }
  }

  if (!ok) {
    return false;
  }

  if (!language.empty()) {
    if (language != "auto" && language != "zh" && language != "en" &&
        language != "ja" && language != "ko" && language != "yue") {
      SHERPA_NCNN_LOGE(
          "Invalid --sense-voice-language: '%s'. Valid values are: auto, zh, "
          "en, "
          "ja, ko, yue. Or you can leave it empty to use 'auto'",
          language.c_str());

      return false;
    }
  }

  return true;
}

std::string OfflineSenseVoiceModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSenseVoiceModelConfig(";
  os << "model_dir=\"" << model_dir << "\", ";
  os << "language=\"" << language << "\", ";
  os << "use_itn=" << (use_itn ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa_ncnn
