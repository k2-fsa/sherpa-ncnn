// sherpa-ncnn/csrc/offline-tts-vits-model.h
//
// Copyright 2025  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_MODEL_H_
#define SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_MODEL_H_

#include "mat.h"  // NOLINT
#include "sherpa-ncnn/csrc/offline-tts-model-config.h"
#include "sherpa-ncnn/csrc/offline-tts-vits-model-meta-data.h"

namespace sherpa_ncnn {

class OfflineTtsVitsModel {
 public:
  ~OfflineTtsVitsModel();
  explicit OfflineTtsVitsModel(const OfflineTtsModelConfig &config);

  const OfflineTtsVitsModelMetaData &GetMetaData() const;

  /**
   * @param x A 2-D tensor of shape (1, num_tokens). Note sequence.w ==
   *          num_tokens
   * @returns Return a vector with 3 tensors: x, m_p, logs_p
   */
  std::vector<ncnn::Mat> RunEncoder(const ncnn::Mat &sequence) const;

  /**
   * @param x It is the x returned by RunEncoder()
   * @param noise A 2-D tensor of shape (2, x.w). Note x.w == noise.w
   *
   * @returns Return logw
   */
  ncnn::Mat RunDurationPredictor(const ncnn::Mat &x,
                                 const ncnn::Mat &noise) const;

  /**
   * @param logw It is returned by RunDurationPredictor()
   * @param m_p It is returned by RunEncoder()
   * @param logs_p It is returned by RunEncoder()
   * @param noise_scale
   * @param speed Note speed = 1 / length_scale, so speed should > 0
   *
   * @returns Return z_p
   */
  static ncnn::Mat PathAttention(const ncnn::Mat &logw, const ncnn::Mat &m_p,
                                 ncnn::Mat &logs_p, float noise_scale,
                                 float speed);

  /**
   * @param z_p It is returned by PathAttention()
   * @returns Return z
   */
  ncnn::Mat RunFlow(const ncnn::Mat &z_p) const;

  /**
   * @param z It is returned by RunFlow()
   * @returns Return a 1-D float tensor containing audio samples
   */
  ncnn::Mat RunDecoder(const ncnn::Mat &z) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_MODEL_H_
