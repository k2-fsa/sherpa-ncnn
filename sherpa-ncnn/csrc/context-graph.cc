// sherpa-ncnn/csrc/context-graph.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-ncnn/csrc/context-graph.h"

#include <cassert>
#include <queue>
#include <utility>

namespace sherpa_ncnn {
void ContextGraph::Build(const std::vector<ContextItem> &token_ids) const {
  for (int32_t i = 0; i < token_ids.size(); ++i) {
    auto node = root_.get();
    auto &ids = std::get<0>(token_ids[i]);
    float score = std::get<1>(token_ids[i]);
    float token_score = score == 0.0 ? context_score_ : score / ids.size();
    for (int32_t j = 0; j < ids.size(); ++j) {
      int32_t token = ids[j];
      bool is_end = j == ids.size() - 1;
      if (0 == node->next.count(token)) {
        node->next[token] = std::make_unique<ContextState>(
            token, token_score, node->node_score + token_score,
            is_end ? node->node_score + token_score : 0, is_end);
      } else {
        ContextState *current_node = node->next[token].get();
        current_node->is_end = is_end || current_node->is_end;
        current_node->token_score =
            std::max(token_score, current_node->token_score);
        current_node->node_score = node->node_score + current_node->token_score;
        current_node->output_score =
            current_node->is_end ? current_node->node_score : 0;
      }
      node = node->next[token].get();
    }
  }
  FillFailOutput();
}

std::pair<float, const ContextState *> ContextGraph::ForwardOneStep(
    const ContextState *state, int32_t token) const {
  const ContextState *node;
  float score;
  if (1 == state->next.count(token)) {
    node = state->next.at(token).get();
    score = node->token_score;
  } else {
    node = state->fail;
    while (0 == node->next.count(token)) {
      node = node->fail;
      if (-1 == node->token) break;  // root
    }
    if (1 == node->next.count(token)) {
      node = node->next.at(token).get();
    }
    score = node->node_score - state->node_score;
  }
  return std::make_pair(score + node->output_score, node);
}

std::pair<float, const ContextState *> ContextGraph::Finalize(
    const ContextState *state) const {
  float score = -state->node_score;
  return std::make_pair(score, root_.get());
}

void ContextGraph::FillFailOutput() const {
  std::queue<const ContextState *> node_queue;
  for (auto &kv : root_->next) {
    kv.second->fail = root_.get();
    node_queue.push(kv.second.get());
  }
  while (!node_queue.empty()) {
    auto current_node = node_queue.front();
    node_queue.pop();
    for (auto &kv : current_node->next) {
      auto fail = current_node->fail;
      if (1 == fail->next.count(kv.first)) {
        fail = fail->next.at(kv.first).get();
      } else {
        fail = fail->fail;
        while (0 == fail->next.count(kv.first)) {
          fail = fail->fail;
          if (-1 == fail->token) break;
        }
        if (1 == fail->next.count(kv.first))
          fail = fail->next.at(kv.first).get();
      }
      kv.second->fail = fail;
      // fill the output arc
      auto output = fail;
      while (!output->is_end) {
        output = output->fail;
        if (-1 == output->token) {
          output = nullptr;
          break;
        }
      }
      kv.second->output = output;
      kv.second->output_score += output == nullptr ? 0 : output->output_score;
      node_queue.push(kv.second.get());
    }
  }
}
}  // namespace sherpa_ncnn
