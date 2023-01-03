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

#ifndef SHERPA_NCNN_CSRC_MODIFIED_BEAM_SEARCH_DECODER_H_
#define SHERPA_NCNN_CSRC_MODIFIED_BEAM_SEARCH_DECODER_H_

#include <memory>
#include <vector>

#include "sherpa-ncnn/csrc/decoder.h"
#include "sherpa-ncnn/csrc/features.h"

namespace sherpa_ncnn {

class ModifiedBeamSearchDecoder : public Decoder {
 public:
  ModifiedBeamSearchDecoder(const DecoderConfig &config,
      std::shared_ptr<Model> model,
      const sherpa_ncnn::SymbolTable &sym,
      std::shared_ptr<Endpoint> endpoint)
     : config_(config),
     model_(model),
     sym_(sym),
     blank_id_(model_->BlankId()),
     context_size_(model_->ContextSize()),
     segment_(model->Segment()),
     offset_(model_->Offset()),
     decoder_input_(context_size_),
     num_processed_(0),
     endpoint_start_frame_(0),
     endpoint_(endpoint) {
       ResetResult();
       BuildDecoderInput((*result_.hyps.begin()).second);
       decoder_out_ = model_->RunDecoder(decoder_input_);
     }

  void AcceptWaveform(int32_t sample_rate,
      const float *input_buffer,
      int32_t frames_per_buffer = 0) override;

  void Decode() override;

  RecognitionResult GetResult() override;

  void ResetResult() override;

  bool IsEndpoint() const override;

 private:
  void BuildDecoderInput(Hypothesis hyp);

  const DecoderConfig &config_;
  sherpa_ncnn::FeatureExtractor feature_extractor_;
  std::shared_ptr<Model> model_;
  const sherpa_ncnn::SymbolTable &sym_;
  const int32_t blank_id_;
  const int32_t context_size_;
  const int32_t segment_;
  const int32_t offset_;
  ncnn::Mat encoder_out_;
  std::vector<ncnn::Mat> encoder_state_;
  ncnn::Mat decoder_input_;
  ncnn::Mat decoder_out_;
  int32_t num_processed_;
  int32_t endpoint_start_frame_;
  std::shared_ptr<Endpoint> endpoint_;
  RecognitionResult result_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_MODIFIED_BEAM_SEARCH_DECODER_H_
