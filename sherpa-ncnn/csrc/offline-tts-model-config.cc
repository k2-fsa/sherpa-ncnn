// sherpa-ncnn/csrc/offline-tts-model-config.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-tts-model-config.h"

#include "sherpa-ncnn/csrc/macros.h"

namespace sherpa_ncnn {

void OfflineTtsModelConfig::Register(ParseOptions *po) {
  vits.Register(po);

  po->Register("num-threads", &num_threads,
               "Number of threads to run the neural network");

  po->Register("debug", &debug,
               "true to print model information while loading it.");
}

bool OfflineTtsModelConfig::Validate() const {
  if (num_threads < 1) {
    SHERPA_NCNN_LOGE("num_threads should be > 0. Given %d", num_threads);
    return false;
  }

  if (!vits.model_dir.empty()) {
    return vits.Validate();
  }

  SHERPA_NCNN_LOGE("Please provide exactly one tts model.");

  return false;
}

std::string OfflineTtsModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsModelConfig(";
  os << "vits=" << vits.ToString() << ", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False");

  return os.str();
}

}  // namespace sherpa_ncnn
