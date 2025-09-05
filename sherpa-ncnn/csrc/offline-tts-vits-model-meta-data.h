// sherpa-ncnn/csrc/offline-tts-vits-model-meta-data.h
//
// Copyright 2025  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_MODEL_META_DATA_H_
#define SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_MODEL_META_DATA_H_

#include <cstdint>
#include <string>
#include <unordered_map>

namespace sherpa_ncnn {

struct OfflineTtsVitsModelMetaData {
  int32_t num_speakers;
  int32_t sample_rate;
  std::unordered_map<std::string, int32_t> token2id;

  // cmn for Chinese
  std::string voice;

  int32_t pad;
  int32_t bos;
  int32_t eos;
};

OfflineTtsVitsModelMetaData ReadFromConfigJson(const std::string& filename);

}  // namespace sherpa_ncnn
#endif  // SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_MODEL_META_DATA_H_
