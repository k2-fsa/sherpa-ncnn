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

#include <memory>
#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/decoder.h"
#include "sherpa-ncnn/csrc/greedy-search-decoder.h"
#include "sherpa-ncnn/csrc/modified-beam-search-decoder.h"

namespace sherpa_ncnn {

Recognizer::Recognizer(const DecoderConfig &decoder_conf,
    const ModelConfig &model_conf,
    const knf::FbankOptions &fbank_opts)
  : decoder_conf_(decoder_conf),
  model_(Model::Create(model_conf)),
  sym_(model_conf.tokens_fn),
  endpoint_(std::make_unique<Endpoint>(decoder_conf.endpoint_config)) {
    if (decoder_conf.method == "modified_beam_search") {
      decoder_ = std::make_unique<ModifiedBeamSearchDecoder>(decoder_conf_,
          model_,
          fbank_opts,
          sym_,
          endpoint_);
    } else {
      decoder_ = std::make_unique<GreedySearchDecoder>(decoder_conf_,
          model_,
          fbank_opts,
          sym_,
          endpoint_);
    }
}

void Recognizer::AcceptWaveform(int32_t sample_rate,
    const float *input_buffer,
    int32_t frames_per_buffer) {
  decoder_->AcceptWaveform(sample_rate, input_buffer, frames_per_buffer);
}

void Recognizer::Decode() {
  decoder_->Decode();
}

RecognitionResult Recognizer::GetResult() {
  return decoder_->GetResult();
}

bool Recognizer::IsEndpoint() const {
  return decoder_->IsEndpoint();
}

}  // namespace sherpa_ncnn
