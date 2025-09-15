// sherpa-ncnn/csrc/offline-stream.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_OFFLINE_STREAM_H_
#define SHERPA_NCNN_CSRC_OFFLINE_STREAM_H_
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "math.h"  // NOLINT
#include "sherpa-ncnn/csrc/features.h"
#include "sherpa-ncnn/csrc/parse-options.h"

namespace sherpa_ncnn {

struct OfflineRecognizerResult {
  // Recognition results.
  // For English, it consists of space separated words.
  // For Chinese, it consists of Chinese words without spaces.
  std::string text;

  // Decoded results at the token level.
  // For instance, for BPE-based models it consists of a list of BPE tokens.
  std::vector<std::string> tokens;

  // for sense-voice only
  std::string lang;

  // for sense-voice only; emotion target of the audio.
  std::string emotion;

  // for sense-voice only; event target of the audio.
  std::string event;

  /// timestamps.size() == tokens.size()
  /// timestamps[i] records the time in seconds when tokens[i] is decoded.
  std::vector<float> timestamps;

  std::string AsJsonString() const;
};

class OfflineStream {
 public:
  explicit OfflineStream(const FeatureExtractorConfig &config = {});
  ~OfflineStream();

  /**
     @param sampling_rate The sampling_rate of the input waveform. If it does
                          not equal to  config.sampling_rate, we will do
                          resampling inside.
     @param waveform Pointer to a 1-D array of size n. It must be normalized to
                     the range [-1, 1].
     @param n Number of entries in waveform

     Caution: You can only invoke this function once so you have to input
              all the samples at once
   */
  void AcceptWaveform(int32_t sampling_rate, const float *waveform,
                      int32_t n) const;

  /// Return feature dim of this extractor.
  ///
  /// Note: if it is Moonshine, then it returns the number of audio samples
  /// currently received.
  int32_t FeatureDim() const;

  /** Get all the feature frames of this stream in a 2-D array
   * @return Return a 2-D tensor of shape (n, feature_dim).
   */
  ncnn::Mat GetFrames() const;

  /** Set the recognition result for this stream. */
  void SetResult(const OfflineRecognizerResult &r);

  /** Get the recognition result of this stream */
  const OfflineRecognizerResult &GetResult() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_STREAM_H_
