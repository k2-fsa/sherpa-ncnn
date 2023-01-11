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
#include "sherpa-ncnn/csrc/endpoint.h"

#include <sstream>
#include <string>

namespace sherpa_ncnn {

static bool RuleActivated(const EndpointRule &rule,
                          const std::string &rule_name,
                          const float trailing_silence,
                          const float utterance_length) {
  bool contain_nonsilence = utterance_length > trailing_silence;
  bool ans = (contain_nonsilence || !rule.must_contain_nonsilence) &&
             trailing_silence >= rule.min_trailing_silence &&
             utterance_length >= rule.min_utterance_length;
  return ans;
}

std::string EndpointRule::ToString() const {
  std::ostringstream os;

  os << "EndpointRule(";
  os << "must_contain_nonsilence="
     << (must_contain_nonsilence ? "True" : "False") << ", ";
  os << "min_trailing_silence=" << min_trailing_silence << ", ";
  os << "min_utterance_length=" << min_utterance_length << ")";

  return os.str();
}

std::string EndpointConfig::ToString() const {
  std::ostringstream os;

  os << "EndpointConfig(";
  os << "rule1=" << rule1.ToString() << ", ";
  os << "rule2=" << rule2.ToString() << ", ";
  os << "rule3=" << rule3.ToString() << ")";

  return os.str();
}

bool Endpoint::IsEndpoint(const int num_frames_decoded,
                          const int trailing_silence_frames,
                          const float frame_shift_in_seconds) const {
  float utterance_length = num_frames_decoded * frame_shift_in_seconds;
  float trailing_silence = trailing_silence_frames * frame_shift_in_seconds;
  if (RuleActivated(config_.rule1, "rule1", trailing_silence,
                    utterance_length) ||
      RuleActivated(config_.rule2, "rule2", trailing_silence,
                    utterance_length) ||
      RuleActivated(config_.rule3, "rule3", trailing_silence,
                    utterance_length)) {
    return true;
  }
  return false;
}

}  // namespace sherpa_ncnn
