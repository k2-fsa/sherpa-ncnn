/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 * Copyright (c)  2022                     (Pingfeng Luo)
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
#include <string>
#include <vector>

#include "mat.h"  // NOLINT
#include "sherpa-ncnn/csrc/hypothesis.h"

namespace sherpa_ncnn {

struct DecoderConfig {
  // supported values are: modified_beam_search, greedy_search
  std::string method = "greedy_search";

  int32_t num_active_paths = 4;  // only used by modified beam search

  DecoderConfig() = default;

  DecoderConfig(const std::string &method, int32_t num_active_paths)
      : method(method), num_active_paths(num_active_paths) {}

  std::string ToString() const;
};

struct DecoderResult {
  /// Number of frames we have decoded so far, counted after subsampling
  int32_t frame_offset = 0;

  /// The decoded token IDs so far
  std::vector<int32_t> tokens;

  /// number of trailing blank frames decoded so far
  int32_t num_trailing_blanks = 0;

  std::vector<int32_t> timestamps;

  // Cache the decoder_out just before endpointing
  ncnn::Mat decoder_out;

  // used only for modified_beam_search
  Hypotheses hyps;
};

class Stream;

class Decoder {
 public:
  virtual ~Decoder() = default;

  /* Return an empty result.
   *
   * To simplify the decoding code, we add `context_size` blanks
   * to the beginning of the decoding result, which will be
   * stripped by calling `StripPrecedingBlanks()`.
   */
  virtual DecoderResult GetEmptyResult() const = 0;

  /** Strip blanks added by `GetEmptyResult()`.
   *
   * @param r It is changed in-place.
   */
  virtual void StripLeadingBlanks(DecoderResult * /*r*/) const {}

  /** Run transducer beam search given the output from the encoder model.
   *
   * @param encoder_out A 3-D tensor of shape (N, T, joiner_dim)
   * @param result  It is modified in-place.
   *
   * @note There is no need to pass encoder_out_length here since for the
   * online decoding case, each utterance has the same number of frames
   * and there are no paddings.
   */
  virtual void Decode(ncnn::Mat encoder_out, DecoderResult *result) = 0;

  virtual void Decode(ncnn::Mat encoder_out, Stream *s, DecoderResult *result) {
    NCNN_LOGE("Please override it!");
    exit(-1);
  }
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_DECODER_H_
