// sherpa-ncnn/csrc/offline-sense-voice-model.h
//
// Copyright (c)  2025  Xiaomi Corporation
#ifndef SHERPA_NCNN_CSRC_OFFLINE_SENSE_VOICE_MODEL_H_
#define SHERPA_NCNN_CSRC_OFFLINE_SENSE_VOICE_MODEL_H_

#include <memory>
#include <vector>

#include "mat.h"  // NOLINT
#include "sherpa-ncnn/csrc/offline-model-config.h"
#include "sherpa-ncnn/csrc/offline-sense-voice-model-meta-data.h"

namespace sherpa_ncnn {

class OfflineSenseVoiceModel {
 public:
  explicit OfflineSenseVoiceModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineSenseVoiceModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineSenseVoiceModel();

  /** Run the forward method of the model.
   *
   * @param features  A tensor of shape (N, T, C). It is changed in-place.
   * @param features_length  A 1-D tensor of shape (N,) containing number of
   *                         valid frames in `features` before padding.
   *                         Its dtype is int32_t.
   * @param language A 1-D tensor of shape (N,) with dtype int32_t
   * @param text_norm A 1-D tensor of shape (N,) with dtype int32_t
   *
   * @return Return logits of shape (N, T, C) with dtype float
   *
   * Note: The subsampling factor is 1 for SenseVoice, so there is
   *       no need to output logits_length.
   */
  ncnn::Mat Forward(const ncnn::Mat &features, int32_t language,
                    int32_t text_norm) const;

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_SENSE_VOICE_MODEL_H_
