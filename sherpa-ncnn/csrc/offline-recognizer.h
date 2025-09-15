// sherpa-ncnn/csrc/offline-recognizer.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_OFFLINE_RECOGNIZER_H_
#define SHERPA_NCNN_CSRC_OFFLINE_RECOGNIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/features.h"
#include "sherpa-ncnn/csrc/offline-model-config.h"
#include "sherpa-ncnn/csrc/offline-stream.h"
#include "sherpa-ncnn/csrc/parse-options.h"

namespace sherpa_ncnn {

struct OfflineRecognizerConfig {
  FeatureExtractorConfig feat_config;
  OfflineModelConfig model_config;

  std::string decoding_method = "greedy_search";

  float blank_penalty = 0.0;

  OfflineRecognizerConfig() = default;
  OfflineRecognizerConfig(const FeatureExtractorConfig &feat_config,
                          const OfflineModelConfig &model_config,
                          const std::string &decoding_method,
                          float blank_penalty)
      : feat_config(feat_config),
        model_config(model_config),
        decoding_method(decoding_method),
        blank_penalty(blank_penalty) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class OfflineRecognizerImpl;

class OfflineRecognizer {
 public:
  ~OfflineRecognizer();

  template <typename Manager>
  OfflineRecognizer(Manager *mgr, const OfflineRecognizerConfig &config);

  explicit OfflineRecognizer(const OfflineRecognizerConfig &config);

  /// Create a stream for decoding.
  std::unique_ptr<OfflineStream> CreateStream() const;

  /** Decode a single stream
   *
   * @param s The stream to decode.
   */
  void DecodeStream(OfflineStream *s) const {
    OfflineStream *ss[1] = {s};
    DecodeStreams(ss, 1);
  }

  /** Decode a list of streams.
   *
   * @param ss Pointer to an array of streams.
   * @param n  Size of the input array.
   */
  void DecodeStreams(OfflineStream **ss, int32_t n) const;

  /**
   * The exact behavior can be defined by a specific recognizer impl.
   */
  void SetConfig(const OfflineRecognizerConfig &config);

  OfflineRecognizerConfig GetConfig() const;

 private:
  std::unique_ptr<OfflineRecognizerImpl> impl_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_RECOGNIZER_H_
