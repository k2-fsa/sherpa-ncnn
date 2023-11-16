// sherpa-ncnn/csrc/context-graph.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_CONTEXT_GRAPH_H_
#define SHERPA_NCNN_CSRC_CONTEXT_GRAPH_H_

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sherpa_ncnn {

class ContextGraph;
using ContextGraphPtr = std::shared_ptr<ContextGraph>;
using ContextItem = std::pair<std::vector<int32_t>, float>;

struct ContextState {
  int32_t token;
  float token_score;
  float node_score;
  float output_score;
  bool is_end;
  std::unordered_map<int32_t, std::unique_ptr<ContextState>> next;
  const ContextState *fail = nullptr;
  const ContextState *output = nullptr;

  ContextState() = default;
  ContextState(int32_t token, float token_score, float node_score,
               float output_score, bool is_end)
      : token(token),
        token_score(token_score),
        node_score(node_score),
        output_score(output_score),
        is_end(is_end) {}
};

class ContextGraph {
 public:
  ContextGraph() = default;
  ContextGraph(const std::vector<ContextItem> &token_ids, float hotwords_score)
      : context_score_(hotwords_score) {
    root_ = std::make_unique<ContextState>(-1, 0, 0, 0, false);
    root_->fail = root_.get();
    Build(token_ids);
  }

  std::pair<float, const ContextState *> ForwardOneStep(
      const ContextState *state, int32_t token_id) const;
  std::pair<float, const ContextState *> Finalize(
      const ContextState *state) const;

  const ContextState *Root() const { return root_.get(); }

 private:
  float context_score_;
  std::unique_ptr<ContextState> root_;
  void Build(const std::vector<ContextItem> &token_ids) const;
  void FillFailOutput() const;
};

}  // namespace sherpa_ncnn
#endif  // SHERPA_NCNN_CSRC_CONTEXT_GRAPH_H_
