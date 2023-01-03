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

#include <memory>
#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/endpoint.h"
#include "sherpa-ncnn/csrc/hypothesis.h"
#include "sherpa-ncnn/csrc/model.h"
#include "sherpa-ncnn/csrc/symbol-table.h"

namespace sherpa_ncnn {

struct RecognitionResult {
  std::vector<int32_t> tokens;
  std::string text;

  int32_t num_trailing_blanks = 0;

  // used only for modified_beam_search
  Hypotheses hyps;
};

struct DecoderConfig {
  std::string method = "greedy_search";

  int32_t num_active_paths = 4;  // for modified beam search
                                 //
  EndpointConfig endpoint_config;
};

class Decoder {
 public:
  Decoder(const DecoderConfig &decoder_conf,
      std::unique_ptr<Model> model,
      const sherpa_ncnn::SymbolTable &sym,
      std::shared_ptr<Endpoint> endpoint);

  Decoder() = default;

  virtual ~Decoder() = default;

  virtual void AcceptWaveform(int32_t sample_rate,
      const float *input_buffer,
      int32_t frames_per_buffer) = 0;

  virtual void Decode() = 0;

  virtual RecognitionResult GetResult() = 0;

  virtual void ResetResult() = 0;

  virtual bool IsEndpoint() const = 0;
};

class Recognizer {
 public:
  /** Construct an instance of OnlineRecognizer.
  */
  Recognizer(const DecoderConfig &decoder_conf,
      const ModelConfig &model_conf);

  ~Recognizer() = default;

  void AcceptWaveform(int32_t sample_rate,
      const float *input_buffer,
      int32_t frames_per_buffer);

  void Decode();

  RecognitionResult GetResult();

  bool IsEndpoint() const;

 private:
  const DecoderConfig &decoder_conf_;
  std::shared_ptr<Model> model_;
  SymbolTable sym_;
  std::shared_ptr<Endpoint> endpoint_;
  std::unique_ptr<Decoder> decoder_;
};

}  // namespace sherpa_ncnn
#endif  // SHERPA_NCNN_CSRC_DECODER_H_
