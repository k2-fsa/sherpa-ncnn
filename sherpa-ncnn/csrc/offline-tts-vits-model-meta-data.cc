// sherpa-ncnn/csrc/offline-tts-vits-model-meta-data.cc
//
// Copyright 2025  Xiaomi Corporation
#include "sherpa-ncnn/csrc/offline-tts-vits-model-meta-data.h"

#include <fstream>
#include <iostream>

#include "nlohmann/json.hpp"

namespace sherpa_ncnn {

OfflineTtsVitsModelMetaData ReadFromConfigJson(const std::string& filename) {
  using json = nlohmann::json;
  std::ifstream f(filename);
  json data = json::parse(f);
  std::cout << data["audio"]["sample_rate"] << "\n";

  OfflineTtsVitsModelMetaData ans;
  ans.sample_rate = data["audio"]["sample_rate"];
  return ans;
}

}  // namespace sherpa_ncnn
