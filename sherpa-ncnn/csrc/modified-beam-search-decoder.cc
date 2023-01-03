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
#include <string>
#include <utility>

#include "sherpa-ncnn/csrc/math.h"
#include "sherpa-ncnn/csrc/modified-beam-search-decoder.h"

namespace sherpa_ncnn {

void ModifiedBeamSearchDecoder::AcceptWaveform(const int32_t sample_rate,
      const float *input_buffer,
      int32_t frames_per_buffer) {
  feature_extractor_.AcceptWaveform(
      sample_rate, input_buffer, frames_per_buffer);
}

void ModifiedBeamSearchDecoder::BuildDecoderInput(Hypothesis hyp) {
  decoder_input_.reshape(context_size_);
  for (int32_t i = 0; i != context_size_; ++i) {
    static_cast<int32_t *>(decoder_input_)[i] =
      *(hyp.ys.end() - context_size_ + i);
  }
}

void ModifiedBeamSearchDecoder::ResetResult() {
  result_.text.clear();
  std::vector<int32_t> blanks(context_size_, blank_id_);
  Hypotheses blank_hyp(std::vector<Hypothesis>(config_.num_active_paths,
        {blanks, 0}));
  result_.hyps = std::move(blank_hyp);
}

void ModifiedBeamSearchDecoder::Decode() {
  while (feature_extractor_.NumFramesReady() - num_processed_ >= segment_) {
    ncnn::Mat features = feature_extractor_.GetFrames(num_processed_, segment_);
    std::tie(encoder_out_, encoder_state_) = model_->RunEncoder(features,
        encoder_state_);

    Hypotheses cur = std::move(result_.hyps);
    std::vector<Hypothesis> prev;
    /* encoder_out_.w == encoder_out_dim, encoder_out_.h == num_frames. */
    for (int32_t t = 0; t != encoder_out_.h; ++t) {
      prev.clear();
      prev.reserve(config_.num_active_paths);
      for (auto &h : cur) {
        prev.push_back(std::move(h.second));
      }
      cur.clear();

      for (auto &h : prev) {
        ncnn::Mat encoder_out_t(encoder_out_.w, encoder_out_.row(t));
        BuildDecoderInput(h);
        decoder_out_ = model_->RunDecoder(decoder_input_);
        ncnn::Mat joiner_out = model_->RunJoiner(encoder_out_t, decoder_out_);
        float *joiner_out_ptr = joiner_out;
        log_softmax(joiner_out_ptr, joiner_out.w);

        auto y = static_cast<int32_t>(std::distance(
              joiner_out_ptr,
              std::max_element(
                joiner_out_ptr,
                joiner_out_ptr + joiner_out.w)));

        if (y != blank_id_) {
          h.ys.push_back(y);
          h.num_trailing_blanks = 0;
        } else {
          h.num_trailing_blanks += 1;
        }
        h.log_prob = LogAdd<double>()(h.log_prob,
            static_cast<double>(joiner_out_ptr[y]));

        // update top num_active_paths
        if (cur.Size() >= config_.num_active_paths) {
          auto cur_least_h = cur.GetLeastProbable(true);
          if (h.log_prob > cur_least_h.log_prob) {
            cur.Remove(cur_least_h);
          }
        }
        cur.Add(std::move(h));
      }
    }

    num_processed_ += offset_;
    result_.hyps = std::move(cur);
  }
}

RecognitionResult ModifiedBeamSearchDecoder::GetResult() {
  // return best result
  auto best_hyp = result_.hyps.GetMostProbable(true);
  std::string best_hyp_text;
  for (const auto & y : best_hyp.ys) {
    best_hyp_text += sym_[y];
  }
  result_.text = std::move(best_hyp_text);
  auto ans = result_;
  if (IsEndpoint()) {
    ResetResult();
    endpoint_start_frame_ = num_processed_;
  }
  return ans;
}

bool ModifiedBeamSearchDecoder::IsEndpoint() const {
  return endpoint_->IsEndpoint(
      num_processed_ - endpoint_start_frame_,
      result_.num_trailing_blanks * 4,
      10 / 1000.0);
}

}  // namespace sherpa_ncnn
