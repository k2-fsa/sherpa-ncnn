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

#ifndef SHERPA_NCNN_CSRC_HYPOTHESIS_H_
#define SHERPA_NCNN_CSRC_HYPOTHESIS_H_

#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-ncnn/csrc/context-graph.h"

namespace sherpa_ncnn {

struct Hypothesis {
  // The predicted tokens so far. Newly predicated tokens are appended.
  std::vector<int32_t> ys;

  // timestamps[i] contains the frame number after subsampling
  // on which ys[i] is decoded.
  std::vector<int32_t> timestamps;

  // The total score of ys in log space.
  double log_prob = 0;
  const ContextState *context_state;
  int32_t num_trailing_blanks = 0;

  Hypothesis() = default;
  Hypothesis(const std::vector<int32_t> &ys, double log_prob,
             const ContextState *context_state = nullptr)
      : ys(ys), log_prob(log_prob), context_state(context_state) {}

  // If two Hypotheses have the same `Key`, then they contain
  // the same token sequence.
  std::string Key() const {
    // TODO(fangjun): Use a hash function?
    std::ostringstream os;
    std::string sep = "-";
    for (auto i : ys) {
      os << i << sep;
      sep = "-";
    }
    return os.str();
  }

  // For debugging
  std::string ToString() const {
    std::ostringstream os;
    os << "(" << Key() << ", " << log_prob << ")";
    return os.str();
  }
};

class Hypotheses {
 public:
  Hypotheses() = default;

  explicit Hypotheses(std::vector<Hypothesis> hyps) {
    for (auto &h : hyps) {
      hyps_dict_[h.Key()] = std::move(h);
    }
  }

  explicit Hypotheses(std::unordered_map<std::string, Hypothesis> hyps_dict)
      : hyps_dict_(std::move(hyps_dict)) {}

  // Add hyp to this object. If it already exists, its log_prob
  // is updated with the given hyp using log-sum-exp.
  void Add(Hypothesis hyp);

  // Get the hyp that has the largest log_prob.
  // If length_norm is true, hyp's log_prob is divided by
  // len(hyp.ys) before comparison.
  Hypothesis GetMostProbable(bool length_norm) const;

  // Get the k hyps that have the largest log_prob.
  // If length_norm is true, hyp's log_prob is divided by
  // len(hyp.ys) before comparison.
  std::vector<Hypothesis> GetTopK(int32_t k, bool length_norm) const;

  int32_t Size() const { return hyps_dict_.size(); }

  std::string ToString() const {
    std::ostringstream os;
    for (const auto &p : hyps_dict_) {
      os << p.second.ToString() << "\n";
    }
    return os.str();
  }

  const auto begin() const { return hyps_dict_.begin(); }
  const auto end() const { return hyps_dict_.end(); }
  auto begin() { return hyps_dict_.begin(); }
  auto end() { return hyps_dict_.end(); }

  void Clear() { hyps_dict_.clear(); }

 private:
  // Return a list of hyps contained in this object.
  std::vector<Hypothesis> Vec() const {
    std::vector<Hypothesis> ans;
    ans.reserve(hyps_dict_.size());
    for (const auto &p : hyps_dict_) {
      ans.push_back(p.second);
    }
    return ans;
  }

 private:
  using Map = std ::unordered_map<std::string, Hypothesis>;
  Map hyps_dict_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_HYPOTHESIS_H_
