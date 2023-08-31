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

#ifndef SHERPA_NCNN_CSRC_RECOGNIZER_H_
#define SHERPA_NCNN_CSRC_RECOGNIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/endpoint.h"
#include "sherpa-ncnn/csrc/features.h"
#include "sherpa-ncnn/csrc/hypothesis.h"
#include "sherpa-ncnn/csrc/model.h"
#include "sherpa-ncnn/csrc/stream.h"
#include "sherpa-ncnn/csrc/symbol-table.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

namespace sherpa_ncnn {

struct RecognitionResult {
  std::string text;
  std::vector<float> timestamps;
  std::vector<int32_t> tokens;

  // String based tokens
  std::vector<std::string> stokens;

  std::string ToString() const;
};

struct RecognizerConfig {
  FeatureExtractorConfig feat_config;
  ModelConfig model_config;
  DecoderConfig decoder_config;
  EndpointConfig endpoint_config;
  bool enable_endpoint = false;

  std::string hotwords_file;

  /// used only for modified_beam_search
  float hotwords_score = 1.5;

  RecognizerConfig() = default;

  RecognizerConfig(const FeatureExtractorConfig &feat_config,
                   const ModelConfig &model_config,
                   const DecoderConfig decoder_config,
                   const EndpointConfig &endpoint_config, bool enable_endpoint,
                   const std::string &hotwords_file, float hotwords_score)
      : feat_config(feat_config),
        model_config(model_config),
        decoder_config(decoder_config),
        endpoint_config(endpoint_config),
        enable_endpoint(enable_endpoint),
        hotwords_file(hotwords_file),
        hotwords_score(hotwords_score) {}

  std::string ToString() const;
};

class Recognizer {
 public:
  explicit Recognizer(const RecognizerConfig &config);

#if __ANDROID_API__ >= 9
  Recognizer(AAssetManager *mgr, const RecognizerConfig &config);
#endif

  ~Recognizer();

  /// Create a stream for decoding.
  std::unique_ptr<Stream> CreateStream() const;

  /**
   * Return true if the given stream has enough frames for decoding.
   * Return false otherwise
   */
  bool IsReady(Stream *s) const;

  void DecodeStream(Stream *s) const;

  // Return true if we detect an endpoint for this stream.
  // Note: If this function returns true, you usually want to
  // invoke Reset(s).
  bool IsEndpoint(Stream *s) const;

  // Clear the state of this stream. If IsEndpoint(s) returns true,
  // after calling this function, IsEndpoint(s) will return false
  void Reset(Stream *s) const;

  RecognitionResult GetResult(Stream *s) const;

  // Return the contained model
  //
  // The user should not free it.
  const Model *GetModel() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_RECOGNIZER_H_
