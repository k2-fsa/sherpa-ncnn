// sherpa-ncnn/csrc/offline-tts-vits-model-meta-data.h
//
// Copyright 2025  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_MODEL_META_DATA_H_
#define SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_MODEL_META_DATA_H_

namespace sherpa_ncnn {

struct OfflineTtsVitsModelMetaData {
  int32_t num_speakers;
  int32_t sample_rate;
};

}  // namespace sherpa_ncnn
#endif  // SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_MODEL_META_DATA_H_
