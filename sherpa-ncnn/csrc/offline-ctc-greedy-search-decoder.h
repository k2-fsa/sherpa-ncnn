// sherpa-ncnn/csrc/offline-ctc-greedy-search-decoder.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_OFFLINE_CTC_GREEDY_SEARCH_DECODER_H_
#define SHERPA_NCNN_CSRC_OFFLINE_CTC_GREEDY_SEARCH_DECODER_H_

#include <vector>

#include "sherpa-ncnn/csrc/offline-ctc-decoder.h"

namespace sherpa_ncnn {

class OfflineCtcGreedySearchDecoder : public OfflineCtcDecoder {
 public:
  explicit OfflineCtcGreedySearchDecoder(int32_t blank_id)
      : blank_id_(blank_id) {}

  OfflineCtcDecoderResult Decode(const ncnn::Mat &logits) override;

 private:
  int32_t blank_id_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_CTC_GREEDY_SEARCH_DECODER_H_
