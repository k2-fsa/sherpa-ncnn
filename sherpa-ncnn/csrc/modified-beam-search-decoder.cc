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

#include <algorithm>
#include <string>
#include <utility>

#include "sherpa-ncnn/csrc/math.h"

namespace sherpa_ncnn {

// @param in 1-D tensor of shape (encoder_dim,)
static ncnn::Mat RepeatEncoderOut(ncnn::Mat in, int32_t n) {
  int32_t w = in.w;
  ncnn::Mat out(w, n, sizeof(float));

  const float *in_ptr = in;
  float *out_ptr = out;

  for (int32_t i = 0; i != n; ++i) {
    std::copy(in_ptr, in_ptr + w, out_ptr);
    out_ptr += w;
  }

  return out;
}

// Compute log_softmax in-place.
//
// @param in_out A 2-d tensor
static void LogSoftmax(ncnn::Mat *in_out) {
  int32_t h = in_out->h;
  int32_t w = in_out->w;
  for (int32_t y = 0; y != h; ++y) {
    float *p = in_out->row(y);
    LogSoftmax(p, w);
  }
}

// The decoder model contains an embedding layer, which only supports
// 1-D output.
// This is a wrapper to suuport 2-D decoder output.
static ncnn::Mat RunDecoder2D(Model *model_, ncnn::Mat decoder_input) {
  ncnn::Mat decoder_out;
  int32_t h = decoder_input.h;

  for (int32_t y = 0; y != h; ++y) {
    ncnn::Mat decoder_input_t =
        ncnn::Mat(decoder_input.w, decoder_input.row(y));

    ncnn::Mat tmp = model_->RunDecoder(decoder_input_t);

    if (y == 0) {
      decoder_out = ncnn::Mat(tmp.w, h);
    }

    const float *ptr = tmp;
    float *outptr = decoder_out.row(y);
    std::copy(ptr, ptr + tmp.w, outptr);
  }

  return decoder_out;
}

void ModifiedBeamSearchDecoder::AcceptWaveform(const float sample_rate,
                                               const float *input_buffer,
                                               int32_t frames_per_buffer) {
  feature_extractor_.AcceptWaveform(sample_rate, input_buffer,
                                    frames_per_buffer);
}

ncnn::Mat ModifiedBeamSearchDecoder::BuildDecoderInput(
    const std::vector<Hypothesis> &hyps) {
  int32_t num_hyps = static_cast<int32_t>(hyps.size());

  ncnn::Mat decoder_input(context_size_, num_hyps);
  auto p = static_cast<int32_t *>(decoder_input);

  for (const auto &hyp : hyps) {
    const auto &ys = hyp.ys;
    std::copy(ys.end() - context_size_, ys.end(), p);
    p += context_size_;
  }

  return decoder_input;
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
    ncnn::Mat encoder_out;
    std::tie(encoder_out, encoder_state_) =
        model_->RunEncoder(features, encoder_state_);

    Hypotheses cur = std::move(result_.hyps);
    /* encoder_out.w == encoder_out_dim, encoder_out.h == num_frames. */
    for (int32_t t = 0; t != encoder_out.h; ++t) {
      std::vector<Hypothesis> prev =
          cur.GetTopK(config_.num_active_paths, true);

      cur.Clear();

      ncnn::Mat decoder_input = BuildDecoderInput(prev);
      ncnn::Mat decoder_out = RunDecoder2D(model_, decoder_input);
      // decoder_out.w == decoder_dim
      // decoder_out.h == num_active_paths

      ncnn::Mat encoder_out_t(encoder_out.w, encoder_out.row(t));
      encoder_out_t = RepeatEncoderOut(encoder_out_t, decoder_out.h);

      ncnn::Mat joiner_out = model_->RunJoiner(encoder_out_t, decoder_out);
      // joiner_out.w == vocab_size
      // joiner_out.h == num_active_paths
      LogSoftmax(&joiner_out);
      auto topk =
          TopkIndex(static_cast<float *>(joiner_out),
                    joiner_out.w * joiner_out.h, config_.num_active_paths);

      for (auto i : topk) {
        int32_t hyp_index = i / joiner_out.w;
        int32_t new_token = i % joiner_out.w;

        const float *p = joiner_out.row(hyp_index);

        Hypothesis new_hyp = prev[hyp_index];

        if (new_token != blank_id_) {
          new_hyp.ys.push_back(new_token);
          new_hyp.num_trailing_blanks = 0;
        } else {
          ++new_hyp.num_trailing_blanks;
        }
        new_hyp.log_prob += p[new_token];
        cur.Add(std::move(new_hyp));
      }
    }  // for (int32_t t = 0; t != encoder_out.h; ++t) {

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

  if (config_.enable_endpoint && IsEndpoint()) {
    ResetResult();
    endpoint_start_frame_ = num_processed_;
  }
  return ans;
}

void ModifiedBeamSearchDecoder::InputFinished() {
  feature_extractor_.InputFinished();
}

bool ModifiedBeamSearchDecoder::IsEndpoint() {
  if (!config_.enable_endpoint) return false;

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
