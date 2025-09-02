// sherpa-ncnn/csrc/offline-tts-impl.h
//
// Copyright (c)  2023-2025  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_OFFLINE_TTS_IMPL_H_
#define SHERPA_NCNN_CSRC_OFFLINE_TTS_IMPL_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/offline-tts.h"

namespace sherpa_ncnn {

class OfflineTtsImpl {
 public:
  virtual ~OfflineTtsImpl() = default;

  static std::unique_ptr<OfflineTtsImpl> Create(const OfflineTtsConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineTtsImpl> Create(Manager *mgr,
                                                const OfflineTtsConfig &config);

  virtual GeneratedAudio Generate(const TtsArgs &args,
                                  GeneratedAudioCallback callback = nullptr,
                                  void *callback_arg = nullptr) const = 0;

  // Return the sample rate of the generated audio
  virtual int32_t SampleRate() const = 0;

  // Number of supported speakers.
  // If it supports only a single speaker, then it return 0 or 1.
  virtual int32_t NumSpeakers() const = 0;

  std::vector<int32_t> AddBlank(const std::vector<int32_t> &x,
                                int32_t blank_id = 0) const;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_TTS_IMPL_H_
