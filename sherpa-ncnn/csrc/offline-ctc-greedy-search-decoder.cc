// sherpa-ncnn/csrc/offline-ctc-greedy-search-decoder.h
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-ctc-greedy-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-ncnn/csrc/macros.h"

namespace sherpa_ncnn {

OfflineCtcDecoderResult OfflineCtcGreedySearchDecoder::Decode(
    const ncnn::Mat &logits) {
  int32_t num_frames = logits.h;
  int32_t vocab_size = logits.w;

  OfflineCtcDecoderResult ans;

  int64_t prev_id = -1;

  for (int32_t t = 0; t != num_frames; ++t) {
    const float *p = logits.row(t);
    int32_t y = std::distance(p, std::max_element(p, p + vocab_size));

    if (y != blank_id_ && y != prev_id) {
      ans.tokens.push_back(y);
      ans.timestamps.push_back(t);
    }

    prev_id = y;
  }  // for (int32_t t = 0; ...)

  return ans;
}

}  // namespace sherpa_ncnn
