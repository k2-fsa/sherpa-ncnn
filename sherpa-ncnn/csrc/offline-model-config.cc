// sherpa-ncnn/csrc/offline-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa-ncnn/csrc/offline-model-config.h"

#include <string>

#include "sherpa-ncnn/csrc/file-utils.h"
#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/text-utils.h"

namespace sherpa_ncnn {

void OfflineModelConfig::Register(ParseOptions *po) {
  sense_voice.Register(po);

  po->Register("tokens", &tokens, "Path to tokens.txt");

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");
}

bool OfflineModelConfig::Validate() const {
  if (num_threads < 1) {
    SHERPA_NCNN_LOGE("num_threads should be > 0. Given %d", num_threads);
    return false;
  }

  if (tokens.empty()) {
    SHERPA_NCNN_LOGE("Please provide --tokens");
  }

  if (!FileExists(tokens)) {
    SHERPA_NCNN_LOGE(
        "tokens: '%s' does not exist. Make sure you provide a file, not a "
        "directory",
        tokens.c_str());
    return false;
  }

  return sense_voice.Validate();
}

std::string OfflineModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineModelConfig(";
  os << "sense_voice=" << sense_voice.ToString() << ", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ")";

  return os.str();
}

}  // namespace sherpa_ncnn
