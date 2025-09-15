// sherpa-ncnn/csrc/offline-ctc-decoder.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_OFFLINE_CTC_DECODER_H_
#define SHERPA_NCNN_CSRC_OFFLINE_CTC_DECODER_H_

#include <vector>

#include "mat.h"  // NOLINT

namespace sherpa_ncnn {

struct OfflineCtcDecoderResult {
  /// The decoded token IDs
  std::vector<int32_t> tokens;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  /// Note: The index is after subsampling
  ///
  /// tokens.size() == timestamps.size()
  std::vector<int32_t> timestamps;
};

class OfflineCtcDecoder {
 public:
  virtual ~OfflineCtcDecoder() = default;

  /** Run CTC decoding given the output from the encoder model.
   *
   * @param logits A 2-D tensor of shape (T, vocab_size) containing
   *                  logits.
   *
   * @return Return the decoded result.
   */
  virtual OfflineCtcDecoderResult Decode(const ncnn::Mat &logits) = 0;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_CTC_DECODER_H_
