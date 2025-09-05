// sherpa-ncnn/csrc/offline-tts-vits-impl.h
//
// Copyright (c)  2023-2025  Xiaomi Corporation
#ifndef SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_IMPL_H_
#define SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_IMPL_H_

#include <stdlib.h>

#include <algorithm>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-ncnn/csrc/lexicon.h"
#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/math.h"
#include "sherpa-ncnn/csrc/offline-tts-vits-model.h"
#include "sherpa-ncnn/csrc/text-utils.h"

namespace sherpa_ncnn {

class OfflineTtsVitsImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsVitsImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsVitsModel>(config.model)) {
    lexicon_ =
        std::make_unique<Lexicon>(config_.model.vits.model_dir + "/lexicon.txt",
                                  model_->GetMetaData().token2id);
  }

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  int32_t NumSpeakers() const override {
    return model_->GetMetaData().num_speakers;
  }

  GeneratedAudio Generate(const TtsArgs &_args,
                          GeneratedAudioCallback callback = nullptr,
                          void *callback_arg = nullptr) const override {
    TtsArgs args = _args;
    if (args.text.empty() && args.tokens.empty()) {
      SHERPA_NCNN_LOGE("Both text and tokens are empty.");
      return {};
    }

    if (!args.text.empty() && !args.tokens.empty()) {
      SHERPA_NCNN_LOGE("Both text and tokens are NOT empty.");
      return {};
    }

    if (!args.text.empty()) {
      args.tokens = Convert(args.text);
    }

    const auto &meta_data = model_->GetMetaData();
    int32_t num_speakers = meta_data.num_speakers;

    if ((num_speakers == 1) && (args.sid != 0)) {
      SHERPA_NCNN_LOGE(
          "This is a single-speaker model and supports only sid 0. Given sid: "
          "%d. sid is ignored.",
          args.sid);
    }

    if ((args.sid >= num_speakers) || (args.sid < 0)) {
      SHERPA_NCNN_LOGE(
          "This model contains only %d speakers. sid should be in the range "
          "[%d, %d]. Given: %d. Use sid=0",
          num_speakers, 0, num_speakers - 1, args.sid);

      args.sid = 0;
    }

    std::vector<float> samples;
    bool should_continue = true;
    int32_t processed = 0;
    int32_t total = args.tokens.size();
    for (const auto &tokens : args.tokens) {
      ++processed;

      ncnn::Mat o = Process(tokens, args.sid, args.noise_scale_w,
                            args.noise_scale, args.speed);

      samples.insert(samples.end(), static_cast<const float *>(o),
                     static_cast<const float *>(o) + o.w);

      if (callback) {
        should_continue = callback(static_cast<const float *>(o), o.w,
                                   processed, total, callback_arg);
        // Caution(fangjun): o is freed when the callback returns, so users
        // should copy the data if they want to access the data after
        // the callback returns to avoid segmentation fault.
      }

      if (!should_continue) {
        break;
      }
    }

    GeneratedAudio ans;
    ans.sample_rate = meta_data.sample_rate;
    ans.samples = std::move(samples);

    return ans;
  }

 private:
  ncnn::Mat Process(const std::vector<int32_t> &_tokens, int32_t sid,
                    float noise_scale_w, float noise_scale, float speed) const {
    // add bos, eos, and pad
    const auto &meta = model_->GetMetaData();
    int32_t bos = meta.bos;
    int32_t eos = meta.eos;
    int32_t pad = meta.pad;

    std::vector<int32_t> tokens(2 * _tokens.size() + 2, pad);
    tokens[0] = bos;
    tokens.back() = eos;

    for (int32_t i = 0; i < _tokens.size(); ++i) {
      tokens[2 * i + 2] = _tokens[i];
    }

    ncnn::Mat sequence(tokens.size(), 1);
    std::copy(tokens.begin(), tokens.end(), static_cast<int32_t *>(sequence));

    auto encoder_out = model_->RunEncoder(sequence);
    sequence.release();

    ncnn::Mat noise(encoder_out[0].w, 2);
    RandomVectorFill(static_cast<float *>(noise), noise.w * noise.h, 0,
                     noise_scale_w);

    ncnn::Mat g = model_->RunEmbedding(sid);

    ncnn::Mat logw = model_->RunDurationPredictor(encoder_out[0], noise, g);

    noise.release();
    encoder_out[0].release();

    ncnn::Mat z_p = model_->PathAttention(logw, encoder_out[1], encoder_out[2],
                                          noise_scale, speed);
    encoder_out.clear();
    logw.release();

    ncnn::Mat z = model_->RunFlow(z_p, g);
    z_p.release();

    ncnn::Mat o = model_->RunDecoder(z, g);

    return o;
  }

  std::vector<std::vector<int32_t>> Convert(const std::string &text) const {
    const auto &voice = model_->GetMetaData().voice;
    if (voice == "cmn") {
      return ConvertChinese(text);
    } else {
      return ConvertNonChinese(text);
    }
  }

  // end is inclusive
  static std::string GetWord(const std::vector<std::string> &words,
                             int32_t start, int32_t end) {
    std::string ans;

    if (start >= words.size() || end >= words.size()) {
      return ans;
    }

    for (int32_t i = start; i <= end; ++i) {
      ans += words[i];
    }

    return ans;
  }

  static std::string NormalizeChinesePunctuation(const std::string &input) {
    std::string text = input;

    static const std::unordered_map<std::string, std::string> punct_map = {
        {"，", ","}, {"。", "."}, {"！", "!"}, {"？", "?"},
        {"：", ":"}, {"；", ";"}, {"（", "("}, {"）", ")"},
        {"【", "["}, {"】", "]"}, {"“", "\""}, {"”", "\""},
        {"‘", "'"},  {"’", "'"},  {"《", "<"}, {"》", ">"}};

    for (const auto &kv : punct_map) {
      text = std::regex_replace(text, std::regex(kv.first), kv.second);
    }

    return text;
  }

  std::vector<std::vector<int32_t>> ConvertChinese(
      const std::string &_text) const {
    auto text = NormalizeChinesePunctuation(_text);

    std::vector<std::string> words = SplitUtf8(text);

    const auto &token2id = model_->GetMetaData().token2id;
    std::vector<std::vector<int32_t>> ans;
    std::vector<int32_t> this_sentence;
    std::vector<int32_t> token_ids;

    int32_t num_words = static_cast<int32_t>(words.size());
    int32_t max_len = 10;
    int32_t space = token2id.at(" ");

    for (int32_t i = 0; i < num_words;) {
      int32_t start = i;
      int32_t end = std::min(i + max_len, num_words - 1);

      std::string w;
      while (end > start) {
        auto this_word = GetWord(words, start, end);

        if (lexicon_->Contains(this_word)) {
          i = end + 1;
          w = std::move(this_word);

          break;
        }

        end -= 1;
      }

      if (w.empty()) {
        w = words[i];
        i += 1;
      }

      lexicon_->TokenizeWord(w, &token_ids);

      if (!token_ids.empty()) {
        this_sentence.insert(this_sentence.end(), token_ids.begin(),
                             token_ids.end());
      } else if (token2id.count(w)) {
        this_sentence.push_back(token2id.at(w));
        if (w == ",") {
          this_sentence.push_back(space);
        }

        if (w == "," || w == "." || w == "?" || w == "!" || w == ";") {
          ans.push_back(std::move(this_sentence));
        }
      } else {
        SHERPA_NCNN_LOGE("empty ids for word %s", w.c_str());
      }
    }  // for (int32_t i = 0; i < num_words;)

    if (!this_sentence.empty()) {
      ans.emplace_back(std::move(this_sentence));
    }

    return ans;
  }

  std::vector<std::vector<int32_t>> ConvertNonChinese(
      const std::string &text) const {
    std::vector<std::string> words = SplitUtf8(text);

    const auto &token2id = model_->GetMetaData().token2id;
    std::vector<std::vector<int32_t>> ans;
    std::vector<int32_t> this_sentence;
    std::vector<int32_t> token_ids;

    int32_t space = token2id.at(" ");

    for (const auto &w : words) {
      lexicon_->TokenizeWord(w, &token_ids);
      if (!token_ids.empty()) {
        this_sentence.insert(this_sentence.end(), token_ids.begin(),
                             token_ids.end());

        this_sentence.push_back(space);
      } else if (token2id.count(w)) {
        this_sentence.push_back(token2id.at(w));
        if (w != "-") {
          // handle cases like two-thirds
          this_sentence.push_back(space);
        }

        if (w == "." || w == "?" || w == "!") {
          ans.push_back(std::move(this_sentence));
        }
      } else {
        SHERPA_NCNN_LOGE("empty ids for word %s", w.c_str());
      }
    }

    if (!this_sentence.empty()) {
      ans.push_back(std::move(this_sentence));
    }

    return ans;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsVitsModel> model_;
  std::unique_ptr<Lexicon> lexicon_;
};

}  // namespace sherpa_ncnn
#endif  // SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_IMPL_H_
