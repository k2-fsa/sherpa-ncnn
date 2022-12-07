/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SHERPA_NCNN_CSRC_DECODER_H_
#define SHERPA_NCNN_CSRC_DECODER_H_

#include "sherpa_ncnn/csrc/hypothesis.h"
#include "sherpa_ncnn/csrc/model.h"

namespace sherpa_ncnn {

struct RecognitionResult {
  // For greedy search
  //   (1) GetInitialResult() initializes it to [blank]*context_size
  //   (2) GetFinalResult() will strip the leading [blank]*context_size
  //
  // For modified_beam_search,
  //   (1) GetInitialResult() leaves it as-is.
  //   (2) GetFinalResult() will fill it with the best result from hyps.
  std::vector<int32_t> tokens;

  // used only for modified_beam_search
  Hypotheses hyps;
};

struct DecoderConfig {
  // Supported values at present: greedy_search
  std::string method = "greedy_search";

  int32_t num_active_paths = 4;  // for modified beam search
};

class Decoder {
 public:
  Decoder(const DecoderConfig &config, Model *model);
  virtual ~Decoder() = default;

  /**
   * For greedy search, it initializes `ret.tokens` to `[blank] * context_size`.
   * For modified beam search, it initializes `ret.hyps` to contain a single
   * hypothesis with value `[blank] * context_size`.
   */
  virtual RecognitionResult GetInitialResult() const = 0;

  /**
   * For greedy search, it strips the leading blanks from `r.tokens`.
   * For modified_beam_search, it gets the best result from `hyps` and
   * set `ret.tokens`.
   */
  virtual RecognitionResult GetFinalResult(
      const RecognitionResult &r) const = 0;

  virtual void Decode(ncnn::Mat &encoder_out, RecognitionResult *result) = 0;
};

}  // namespace sherpa_ncnn
#endif  // SHERPA_NCNN_CSRC_DECODER_H_
