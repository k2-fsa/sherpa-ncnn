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

#include "sherpa-ncnn/csrc/recognizer.h"

#include <memory>
#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/greedy-search-decoder.h"
#include "sherpa-ncnn/csrc/modified-beam-search-decoder.h"

namespace sherpa_ncnn {

std::string DecoderConfig::ToString() const {
  std::ostringstream os;

  os << "DecoderConfig(";
  os << "method=\"" << method << "\", ";
  os << "num_active_paths=" << num_active_paths << ", ";
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ", ";
  os << "endpoint_config=" << endpoint_config.ToString() << ")";

  return os.str();
}

Recognizer::Recognizer(
#if __ANDROID_API__ >= 9
    AAssetManager *mgr,
#endif
    const DecoderConfig &decoder_conf, const ModelConfig &model_conf,
    const knf::FbankOptions &fbank_opts)
    :
#if __ANDROID_API__ >= 9
      model_(Model::Create(mgr, model_conf)),
      sym_(std::make_unique<SymbolTable>(mgr, model_conf.tokens)),
#else
      model_(Model::Create(model_conf)),
      sym_(std::make_unique<SymbolTable>(model_conf.tokens)),
#endif
      endpoint_(std::make_unique<Endpoint>(decoder_conf.endpoint_config)) {
  if (decoder_conf.method == "modified_beam_search") {
    decoder_ = std::make_unique<ModifiedBeamSearchDecoder>(
        decoder_conf, model_.get(), fbank_opts, sym_.get(), endpoint_.get());
  } else if (decoder_conf.method == "greedy_search") {
    decoder_ = std::make_unique<GreedySearchDecoder>(
        decoder_conf, model_.get(), fbank_opts, sym_.get(), endpoint_.get());
  } else {
    NCNN_LOGE("Unsupported decoding method: %s\n", decoder_conf.method.c_str());
    exit(-1);
  }
}

void Recognizer::AcceptWaveform(float sample_rate, const float *input_buffer,
                                int32_t frames_per_buffer) {
  decoder_->AcceptWaveform(sample_rate, input_buffer, frames_per_buffer);
}

void Recognizer::Decode() { decoder_->Decode(); }

RecognitionResult Recognizer::GetResult() { return decoder_->GetResult(); }

bool Recognizer::IsEndpoint() { return decoder_->IsEndpoint(); }

void Recognizer::Reset() { return decoder_->Reset(); }

void Recognizer::InputFinished() { return decoder_->InputFinished(); }

}  // namespace sherpa_ncnn
