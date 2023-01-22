/**
 * Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "sherpa-ncnn/c-api/c-api.h"

#include "sherpa-ncnn/csrc/model.h"
#include "sherpa-ncnn/csrc/recognizer.h"

#ifdef __cplusplus
#define SHERPA_NCNN_EXTERN_C extern "C"
#endif

SHERPA_NCNN_EXTERN_C
struct SherpaNcnnRecognizer {
  std::unique_ptr<sherpa_ncnn::Recognizer> recognizer;
};

SherpaNcnnRecognizer *CreateRecognizer(
    const SherpaNcnnModelConfig *in_model_config,
    const SherpaNcnnDecoderConfig *in_decoder_config) {
  // model_config
  sherpa_ncnn::ModelConfig model_config;
  model_config.encoder_param = in_model_config->encoder_param;
  model_config.encoder_bin = in_model_config->encoder_bin;

  model_config.decoder_param = in_model_config->decoder_param;
  model_config.decoder_bin = in_model_config->decoder_bin;

  model_config.joiner_param = in_model_config->joiner_param;
  model_config.joiner_bin = in_model_config->joiner_bin;

  model_config.tokens = in_model_config->tokens;
  model_config.use_vulkan_compute = in_model_config->use_vulkan_compute;

  int32_t num_threads = in_model_config->num_threads;

  model_config.encoder_opt.num_threads = num_threads;
  model_config.decoder_opt.num_threads = num_threads;
  model_config.joiner_opt.num_threads = num_threads;

  // decoder_config
  sherpa_ncnn::DecoderConfig decoder_config;
  decoder_config.method = in_decoder_config->decoding_method;

  decoder_config.enable_endpoint = in_decoder_config->enable_endpoint;

  sherpa_ncnn::EndpointConfig endpoint_config;

  endpoint_config.rule1.min_trailing_silence =
      in_decoder_config->rule1_min_trailing_silence;

  endpoint_config.rule2.min_trailing_silence =
      in_decoder_config->rule2_min_trailing_silence;

  endpoint_config.rule3.min_utterance_length =
      in_decoder_config->rule3_min_utterance_length;

  decoder_config.endpoint_config = endpoint_config;

  float expected_sampling_rate = 16000;
  knf::FbankOptions fbank_opts;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.snip_edges = false;
  fbank_opts.frame_opts.samp_freq = expected_sampling_rate;
  fbank_opts.mel_opts.num_bins = 80;

  auto ans = new SherpaNcnnRecognizer;
  ans->recognizer = std::make_unique<sherpa_ncnn::Recognizer>(
      decoder_config, model_config, fbank_opts);
  return ans;
}

void DestroyRecognizer(SherpaNcnnRecognizer *p) { delete p; }

void AcceptWaveform(SherpaNcnnRecognizer *p, float sample_rate,
                    const float *samples, int32_t n) {
  p->recognizer->AcceptWaveform(sample_rate, samples, n);
}

void Decode(SherpaNcnnRecognizer *p) { p->recognizer->Decode(); }

SherpaNcnnResult *GetResult(SherpaNcnnRecognizer *p) {
  std::string text = p->recognizer->GetResult().text;

  auto r = new SherpaNcnnResult;
  r->text = new char[text.size() + 1];
  std::copy(text.begin(), text.end(), const_cast<char *>(r->text));
  const_cast<char *>(r->text)[text.size()] = 0;

  return r;
}

void DestroyResult(const SherpaNcnnResult *r) {
  delete[] r->text;
  delete r;
}

void Reset(SherpaNcnnRecognizer *p) { p->recognizer->Reset(); }

void InputFinished(SherpaNcnnRecognizer *p) { p->recognizer->InputFinished(); }

int32_t IsEndpoint(SherpaNcnnRecognizer *p) {
  return p->recognizer->IsEndpoint();
}
