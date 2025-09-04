// sherpa-ncnn/csrc/offline-tts-vits-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-tts-vits-model-config.h"

#include <vector>

#include "sherpa-ncnn/csrc/file-utils.h"
#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/offline-tts-vits-model-meta-data.h"

namespace sherpa_ncnn {

void OfflineTtsVitsModelConfig::Register(ParseOptions *po) {
  po->Register("vits-model-dir", &model_dir, "Path to VITS model");
}

bool OfflineTtsVitsModelConfig::Validate() const {
  if (model_dir.empty()) {
    SHERPA_NCNN_LOGE("Please provide --vits-model-dir");
    return false;
  }

  if (!FileExists(model_dir + "/config.json")) {
    SHERPA_NCNN_LOGE("'%s' does not exist!",
                     (model_dir + "/config.json").c_str());
    return false;
  }

  std::vector<std::string> files_to_check = {
      "lexicon.txt",   "encoder.ncnn.param", "encoder.ncnn.bin",
      "dp.ncnn.param", "dp.ncnn.bin",        "flow.ncnn.param",
      "flow.ncnn.bin", "decoder.ncnn.param", "decoder.ncnn.bin",
  };

  OfflineTtsVitsModelMetaData meta =
      ReadFromConfigJson(model_dir + "/config.json");

  if (meta.num_speakers > 1) {
    files_to_check.push_back("embedding.ncnn.param");
    files_to_check.push_back("embedding.ncnn.bin");
  }

  bool ok = true;
  for (const auto &f : files_to_check) {
    auto name = model_dir + "/" + f;
    if (!FileExists(name)) {
      SHERPA_NCNN_LOGE("'%s' does not exist inside the directory '%s'",
                       name.c_str(), model_dir.c_str());
      ok = false;
    }
  }

  return ok;
}

std::string OfflineTtsVitsModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsVitsModelConfig(";
  os << "model_dir=\"" << model_dir << "\")";

  return os.str();
}

}  // namespace sherpa_ncnn
