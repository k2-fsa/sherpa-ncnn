/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include <algorithm>
#include <utility>

#include "sherpa-ncnn/csrc/hypothesis.h"
#include "sherpa-ncnn/csrc/math.h"

namespace sherpa_ncnn {

void Hypotheses::Add(Hypothesis hyp) {
  auto key = hyp.Key();
  auto it = hyps_dict_.find(key);
  if (it == hyps_dict_.end()) {
    hyps_dict_[key] = std::move(hyp);
  } else {
    it->second.log_prob = LogAdd<double>()(it->second.log_prob, hyp.log_prob);
  }
}

Hypothesis Hypotheses::GetMostProbable(bool length_norm) const {
  if (length_norm == false) {
    return std::max_element(hyps_dict_.begin(), hyps_dict_.end(),
                            [](const auto &left, auto &right) -> bool {
                              return left.second.log_prob <
                                     right.second.log_prob;
                            })
        ->second;
  } else {
    // for length_norm is true
    return std::max_element(
               hyps_dict_.begin(), hyps_dict_.end(),
               [](const auto &left, const auto &right) -> bool {
                 return left.second.log_prob / left.second.ys.size() <
                        right.second.log_prob / right.second.ys.size();
               })
        ->second;
  }
}

Hypothesis Hypotheses::GetLeastProbable(bool length_norm) const {
  if (length_norm == false) {
    return std::max_element(hyps_dict_.begin(), hyps_dict_.end(),
                            [](const auto &left, auto &right) -> bool {
                              return left.second.log_prob >
                                     right.second.log_prob;
                            })
        ->second;
  } else {
    // for length_norm is true
    return std::max_element(
               hyps_dict_.begin(), hyps_dict_.end(),
               [](const auto &left, const auto &right) -> bool {
                 return left.second.log_prob / left.second.ys.size() >
                        right.second.log_prob / right.second.ys.size();
               })
        ->second;
  }
}

}  // namespace sherpa_ncnn
