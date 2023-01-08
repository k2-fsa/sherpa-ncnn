/**
 * Copyright      2022  (authors: Pingfeng Luo)
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
#ifndef SHERPA_NCNN_CSRC_ENDPOINT_H_
#define SHERPA_NCNN_CSRC_ENDPOINT_H_

#include <string>
#include <vector>

namespace sherpa_ncnn {

struct EndpointRule {
  // If True, for this endpointing rule to apply there must
  // be nonsilence in the best-path traceback.
  // For decoding, a non-blank token is considered as non-silence
  bool must_contain_nonsilence;
  // This endpointing rule requires duration of trailing silence
  // (in seconds) to be >= this value.
  float min_trailing_silence;
  // This endpointing rule requires utterance-length (in seconds)
  // to be >= this value.
  float min_utterance_length;

  explicit EndpointRule(const bool must_contain_nonsilence = true,
                        const float min_trailing_silence = 2.0,
                        const float min_utterance_length = 0)
      : must_contain_nonsilence(must_contain_nonsilence),
        min_trailing_silence(min_trailing_silence),
        min_utterance_length(min_utterance_length) {}

  std::string ToString() const;
};

struct EndpointConfig {
  // For default setting,
  // rule1 times out after 2.4 seconds of silence, even if we decoded nothing.
  // rule2 times out after 1.2 seconds of silence after decoding something.
  // rule3 times out after the utterance is 20 seconds long, regardless of
  // anything else.
  EndpointRule rule1;
  EndpointRule rule2;
  EndpointRule rule3;

  EndpointConfig(const EndpointRule &rule1, const EndpointRule &rule2,
                 const EndpointRule &rule3)
      : rule1(rule1), rule2(rule2), rule3(rule3) {}

  EndpointConfig()
      : rule1(false, 2.4, 0), rule2(true, 1.4, 0), rule3(false, 0, 20) {}

  std::string ToString() const;
};

class Endpoint {
 public:
  explicit Endpoint(const EndpointConfig &config) : config_(config) {}

  /// This function returns true if this set of endpointing rules thinks we
  /// should terminate decoding.
  bool IsEndpoint(const int num_frames_decoded,
                  const int trailing_silence_frames,
                  const float frame_shift_in_seconds) const;

 private:
  EndpointConfig config_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_ENDPOINT_H_
