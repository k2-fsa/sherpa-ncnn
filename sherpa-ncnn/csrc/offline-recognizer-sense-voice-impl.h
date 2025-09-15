// sherpa-ncnn/csrc/offline-recognizer-sense-voice-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_
#define SHERPA_NCNN_CSRC_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/offline-ctc-greedy-search-decoder.h"
#include "sherpa-ncnn/csrc/offline-model-config.h"
#include "sherpa-ncnn/csrc/offline-recognizer-impl.h"
#include "sherpa-ncnn/csrc/offline-recognizer.h"
#include "sherpa-ncnn/csrc/offline-sense-voice-model.h"
#include "sherpa-ncnn/csrc/offline-stream.h"
#include "sherpa-ncnn/csrc/symbol-table.h"

namespace sherpa_ncnn {

OfflineRecognizerResult ConvertSenseVoiceResult(
    const OfflineCtcDecoderResult &src, const SymbolTable &sym_table,
    int32_t frame_shift_ms, int32_t subsampling_factor) {
  OfflineRecognizerResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());

  std::string text;

  for (int32_t i = 4; i < src.tokens.size(); ++i) {
    auto sym = sym_table[src.tokens[i]];
    text.append(sym);

    r.tokens.push_back(std::move(sym));
  }
  r.text = std::move(text);

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;

  for (int32_t i = 4; i < src.timestamps.size(); ++i) {
    float time = frame_shift_s * (src.timestamps[i] - 4);
    r.timestamps.push_back(time);
  }

  // parse lang, emotion and event from tokens.
  if (src.tokens.size() >= 3) {
    r.lang = sym_table[src.tokens[0]];
    r.emotion = sym_table[src.tokens[1]];
    r.event = sym_table[src.tokens[2]];
  }

  return r;
}

class OfflineRecognizerSenseVoiceImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerSenseVoiceImpl(
      const OfflineRecognizerConfig &config)
      : config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineSenseVoiceModel>(config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerSenseVoiceImpl(Manager *mgr,
                                  const OfflineRecognizerConfig &config)
      : config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineSenseVoiceModel>(mgr,
                                                        config.model_config)) {
    Init();
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i < n; ++i) {
      DecodeOneStream(ss[i]);
    }
  }

  void SetConfig(const OfflineRecognizerConfig &config) override {
    config_ = config;
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void Init() {
    const auto &meta_data = model_->GetModelMetadata();
    if (config_.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OfflineCtcGreedySearchDecoder>(meta_data.blank_id);
    } else {
      SHERPA_NCNN_LOGE("Only greedy_search is supported at present. Given %s",
                       config_.decoding_method.c_str());
      SHERPA_NCNN_EXIT(-1);
    }

    InitFeatConfig();
  }

  void InitFeatConfig() {
    const auto &meta_data = model_->GetModelMetadata();

    config_.feat_config.normalize_samples = meta_data.normalize_samples;
    config_.feat_config.window_type = "hamming";
    config_.feat_config.high_freq = 0;
    config_.feat_config.snip_edges = true;
  }

  void DecodeOneStream(OfflineStream *s) const {
    const auto &meta_data = model_->GetModelMetadata();

    ncnn::Mat f = s->GetFrames();
    f = ApplyLFR(f);

    int32_t language = 0;
    if (config_.model_config.sense_voice.language.empty()) {
      language = 0;
    } else if (meta_data.lang2id.count(
                   config_.model_config.sense_voice.language)) {
      language =
          meta_data.lang2id.at(config_.model_config.sense_voice.language);
    } else {
      SHERPA_NCNN_LOGE("Unknown language: %s. Use 0 instead.",
                       config_.model_config.sense_voice.language.c_str());
    }

    int32_t text_norm = config_.model_config.sense_voice.use_itn
                            ? meta_data.with_itn_id
                            : meta_data.without_itn_id;

    ncnn::Mat logits = model_->Forward(f, language, text_norm);

    auto result = decoder_->Decode(logits);

    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = meta_data.window_shift;
    auto r = ConvertSenseVoiceResult(result, symbol_table_, frame_shift_ms,
                                     subsampling_factor);
    s->SetResult(r);
  }

  ncnn::Mat ApplyLFR(const ncnn::Mat &in) const {
    const auto &meta_data = model_->GetModelMetadata();

    int32_t lfr_window_size = meta_data.window_size;
    int32_t lfr_window_shift = meta_data.window_shift;
    int32_t in_feat_dim = in.w;

    int32_t in_num_frames = in.h;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;
    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    ncnn::Mat out(out_feat_dim, out_num_frames);

    const float *p_in = in;
    float *p_out = out;

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);

      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineSenseVoiceModel> model_;
  std::unique_ptr<OfflineCtcDecoder> decoder_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_
