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

namespace sherpa_ncnn {

void GreedySearchDecoder::AcceptWaveform(const int32_t sample_rate,
      const float *input_buffer,
      int32_t frames_per_buffer) {
  feature_extractor_.AcceptWaveform(
      sample_rate, input_buffer, frames_per_buffer);
}

void GreedySearchDecoder::BuildDecoderInput() {
  decoder_input_.reshape(context_size_);
  for (int32_t i = 0; i != context_size_; ++i) {
    static_cast<int32_t *>(decoder_input_)[i] =
      *(result_.tokens.end() - context_size_ + i);
  }
}

void GreedySearchDecoder::ResetResult() {
  result_.tokens.clear();
  result_.text.clear();
  for (int32_t i = 0; i != context_size_; ++i) {
    result_.tokens.push_back(blank_id_);
  }
}

void GreedySearchDecoder::Decode() {
  while (feature_extractor_.NumFramesReady() - num_processed_ >= segment_) {
    ncnn::Mat features = feature_extractor_.GetFrames(num_processed_, segment_);
    std::tie(encoder_out_, encoder_state_) = model_->RunEncoder(features,
        encoder_state_);

    /* encoder_out_.w == encoder_out_dim, encoder_out_.h == num_frames. */
    for (int32_t t = 0; t != encoder_out_.h; ++t) {
      ncnn::Mat encoder_out_t(encoder_out_.w, encoder_out_.row(t));
      ncnn::Mat joiner_out = model_->RunJoiner(encoder_out_t, decoder_out_);
      float *joiner_out_ptr = joiner_out;

      auto y = static_cast<int32_t>(std::distance(
            joiner_out_ptr,
            std::max_element(
              joiner_out_ptr,
              joiner_out_ptr + joiner_out.w)));

      if (y != blank_id_) {
        result_.tokens.push_back(y);
        result_.text += sym_[y];
        BuildDecoderInput();
        decoder_out_ = model_->RunDecoder(decoder_input_);
        result_.num_trailing_blanks = 0;
      } else {
        ++result_.num_trailing_blanks;
      }
    }

    num_processed_ += offset_;
  }
}

RecognitionResult GreedySearchDecoder::GetResult() {
  auto ans = result_;
  if (IsEndpoint()) {
    ResetResult();
    endpoint_start_frame_ = num_processed_;
  }
  return ans;
}

bool GreedySearchDecoder::IsEndpoint() const {
  return endpoint_->IsEndpoint(
      num_processed_ - endpoint_start_frame_,
      result_.num_trailing_blanks * 4,
      10 / 1000.0);
}

}  // namespace sherpa_ncnn
