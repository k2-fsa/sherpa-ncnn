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
#include "sherpa-ncnn/csrc/modified-beam-search-decoder.h"

#include <string>
#include <utility>

#include "sherpa-ncnn/csrc/math.h"

namespace sherpa_ncnn {

void ModifiedBeamSearchDecoder::AcceptWaveform(const int32_t sample_rate,
                                               const float *input_buffer,
                                               int32_t frames_per_buffer) {
  feature_extractor_.AcceptWaveform(sample_rate, input_buffer,
                                    frames_per_buffer);
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
  Hypotheses blank_hyp({{blanks, 0}});
  result_.hyps = std::move(blank_hyp);
  result_.num_trailing_blanks = 0;
}

void ModifiedBeamSearchDecoder::Decode() {
  while (feature_extractor_.NumFramesReady() - num_processed_ >= segment_) {
    ncnn::Mat features = feature_extractor_.GetFrames(num_processed_, segment_);
    std::tie(encoder_out_, encoder_state_) =
        model_->RunEncoder(features, encoder_state_);

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

      for (const auto &h : prev) {
        ncnn::Mat encoder_out_t(encoder_out_.w, encoder_out_.row(t));
        BuildDecoderInput(h);
        decoder_out_ = model_->RunDecoder(decoder_input_);
        ncnn::Mat joiner_out = model_->RunJoiner(encoder_out_t, decoder_out_);
        auto joiner_out_ptr = joiner_out.row(0);
        log_softmax(joiner_out_ptr, joiner_out.w);

        // update active_paths
        auto topk =
            topk_index(joiner_out_ptr, joiner_out.w, config_.num_active_paths);
        for (int i = 0; i != topk.size(); i++) {
          Hypothesis new_hyp = h;
          int32_t new_token = topk[i];
          if (new_token != blank_id_) {
            new_hyp.ys.push_back(new_token);
            new_hyp.num_trailing_blanks = 0;
          } else {
            new_hyp.num_trailing_blanks += 1;
          }
          new_hyp.log_prob += joiner_out_ptr[new_token];
          cur.Add(std::move(new_hyp));
        }
      }
      while (cur.Size() > config_.num_active_paths) {
        auto least_hyp = cur.GetLeastProbable(true);
        cur.Remove(least_hyp);
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
  for (const auto &token : best_hyp.ys) {
    if (token != blank_id_) {
      best_hyp_text += (*sym_)[token];
    }
  }
  result_.text = std::move(best_hyp_text);
  result_.num_trailing_blanks = best_hyp.num_trailing_blanks;
  auto ans = result_;

  if (config_.use_endpoint && IsEndpoint()) {
    ResetResult();
    endpoint_start_frame_ = num_processed_;
  }
  return ans;
}

void ModifiedBeamSearchDecoder::InputFinished() {
  feature_extractor_.InputFinished();
}

bool ModifiedBeamSearchDecoder::IsEndpoint() {
  auto best_hyp = result_.hyps.GetMostProbable(true);
  result_.num_trailing_blanks = best_hyp.num_trailing_blanks;
  return endpoint_->IsEndpoint(num_processed_ - endpoint_start_frame_,
                               result_.num_trailing_blanks * 4, 10 / 1000.0);
}

void ModifiedBeamSearchDecoder::Reset() {
  ResetResult();
  feature_extractor_.Reset();
  num_processed_ = 0;
  endpoint_start_frame_ = 0;
}

}  // namespace sherpa_ncnn
