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
#include "sherpa-ncnn/csrc/greedy-search-decoder.h"

#include <vector>

namespace sherpa_ncnn {

ncnn::Mat GreedySearchDecoder::BuildDecoderInput(
    const DecoderResult &result) const {
  int32_t context_size = model_->ContextSize();
  ncnn::Mat decoder_input(context_size);
  for (int32_t i = 0; i != context_size; ++i) {
    static_cast<int32_t *>(decoder_input)[i] =
        *(result.tokens.end() - context_size + i);
  }
  return decoder_input;
}

DecoderResult GreedySearchDecoder::GetEmptyResult() const {
  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0
  DecoderResult r;
  r.tokens.resize(context_size, blank_id);

  return r;
}

void GreedySearchDecoder::StripLeadingBlanks(DecoderResult *r) const {
  int32_t context_size = model_->ContextSize();

  auto start = r->tokens.begin() + context_size;
  auto end = r->tokens.end();

  r->tokens = std::vector<int32_t>(start, end);
}

void GreedySearchDecoder::Decode(ncnn::Mat encoder_out, DecoderResult *result) {
  // TODO(fangjun): Cache the result of decoder_out
  ncnn::Mat decoder_out = result->decoder_out;
  if (decoder_out.empty()) {
    ncnn::Mat decoder_input = BuildDecoderInput(*result);
    decoder_out = model_->RunDecoder(decoder_input);
  }

  int32_t frame_offset = result->frame_offset;
  for (int32_t t = 0; t != encoder_out.h; ++t) {
    ncnn::Mat encoder_out_t(encoder_out.w, encoder_out.row(t));
    ncnn::Mat joiner_out = model_->RunJoiner(encoder_out_t, decoder_out);

    const float *joiner_out_ptr = joiner_out.row(0);

    auto new_token = static_cast<int32_t>(std::distance(
        joiner_out_ptr,
        std::max_element(joiner_out_ptr, joiner_out_ptr + joiner_out.w)));

    // the blank ID is fixed to 0
    if (new_token != 0) {
      result->tokens.push_back(new_token);
      ncnn::Mat decoder_input = BuildDecoderInput(*result);
      decoder_out = model_->RunDecoder(decoder_input);
      result->num_trailing_blanks = 0;
      result->timestamps.push_back(t + frame_offset);
    } else {
      ++result->num_trailing_blanks;
    }
  }

  result->frame_offset += encoder_out.h;
  result->decoder_out = decoder_out;
}

}  // namespace sherpa_ncnn
