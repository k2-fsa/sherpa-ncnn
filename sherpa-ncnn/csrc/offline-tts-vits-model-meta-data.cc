// sherpa-ncnn/csrc/offline-tts-vits-model-meta-data.cc
//
// Copyright 2025  Xiaomi Corporation
#include "sherpa-ncnn/csrc/offline-tts-vits-model-meta-data.h"

#include <fstream>

#include "nlohmann/json.hpp"

namespace sherpa_ncnn {

static std::unordered_map<std::string, int32_t> LoadToken2ID(
    const nlohmann::json& data) {
  std::unordered_map<std::string, int32_t> ans;
  for (const auto& it : data["phoneme_id_map"].items()) {
    ans[it.key()] = it.value()[0];
  }

  return ans;
}

OfflineTtsVitsModelMetaData ReadFromConfigJson(const std::string& filename) {
  std::ifstream f(filename);
  auto data = nlohmann::json::parse(f);

  OfflineTtsVitsModelMetaData ans;
  ans.sample_rate = data["audio"]["sample_rate"];
  ans.voice = data["espeak"]["voice"];
  ans.num_speakers = data["num_speakers"];
  ans.token2id = LoadToken2ID(data);

  ans.pad = ans.token2id["_"];
  ans.bos = ans.token2id["^"];
  ans.eos = ans.token2id["$"];

  return ans;
}

}  // namespace sherpa_ncnn
