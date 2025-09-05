// sherpa-ncnn/csrc/offline-tts.h
//
// Copyright (c)  2023-2025  Xiaomi Corporation
#ifndef SHERPA_NCNN_CSRC_OFFLINE_TTS_H_
#define SHERPA_NCNN_CSRC_OFFLINE_TTS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/offline-tts-model-config.h"
#include "sherpa-ncnn/csrc/parse-options.h"

namespace sherpa_ncnn {

struct OfflineTtsConfig {
  OfflineTtsModelConfig model;
  // If not empty, it contains a list of rule FST filenames.
  // Filenames are separated by a comma.
  // Example value: rule1.fst,rule2,fst,rule3.fst
  //
  // If there are multiple rules, they are applied from left to right.
  std::string rule_fsts;

  // If there are multiple FST archives, they are applied from left to right.
  std::string rule_fars;

  // Maximum number of sentences that we process at a time.
  // This is to avoid OOM for very long input text.
  // If you set it to -1, then we process all sentences in a single batch.
  int32_t max_num_sentences = 1;

  // If positive, we limit the max number of tokens per sentence
  int32_t max_tokens_per_sentence = -1;

  // A silence interval contains audio samples with value close to 0.
  //
  // the duration of the new interval is old_duration * silence_scale.
  float silence_scale = 1.0;

  OfflineTtsConfig() = default;
  OfflineTtsConfig(const OfflineTtsModelConfig &model,
                   const std::string &rule_fsts, const std::string &rule_fars,
                   int32_t max_num_sentences, float silence_scale)
      : model(model),
        rule_fsts(rule_fsts),
        rule_fars(rule_fars),
        max_num_sentences(max_num_sentences),
        silence_scale(silence_scale) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

struct GeneratedAudio {
  std::vector<float> samples;
  int32_t sample_rate;

  // Silence means pause here.
  // If scale > 1, then it increases the duration of a pause
  // If scale < 1, then it reduces the duration of a pause
  GeneratedAudio ScaleSilence(float scale) const;
};

struct TtsArgs {
  TtsArgs() = default;
  TtsArgs(const std::string &text,
          const std::vector<std::vector<int32_t>> &tokens, int32_t sid = 0,
          float speed = 1.0, float noise_scale = 0.667f,
          float noise_scale_w = 0.8f)
      : text(text),
        tokens(tokens),
        sid(sid),
        speed(speed),
        noise_scale(noise_scale),
        noise_scale_w(noise_scale_w) {}
  // A string containing words separated by spaces
  std::string text;

  // If not empty, text is ignored we we assumes users have
  // converted text to token IDs. tokens[i] is the token IDs for
  // the i-th sentence.
  std::vector<std::vector<int32_t>> tokens;

  // Used only for multi-speaker models, e.g., models
  // trained using the VCTK dataset. It is not used for
  // single-speaker models, e.g., models trained using the ljspeech
  // dataset.
  int32_t sid = 0;

  // The speed for the generated speech. E.g., 2 means 2x faster.
  float speed = 1.0;  // speed = 1.0/length_scale

  float noise_scale = 0.667f;
  float noise_scale_w = 0.8f;
};

class OfflineTtsImpl;

// If the callback returns 0, then it stops generating
// if the callback returns 1, then it keeps generating
using GeneratedAudioCallback = std::function<int32_t(
    const float * /*samples*/, int32_t /*num_samples*/, int32_t /*processed*/,
    int32_t /*total*/, void * /*arg*/)>;

class OfflineTts {
 public:
  ~OfflineTts();
  explicit OfflineTts(const OfflineTtsConfig &config);

  // @param callback If not NULL, it is called whenever config.max_num_sentences
  //                 sentences have been processed. Note that the passed
  //                 pointer `samples` for the callback might be invalidated
  //                 after the callback is returned, so the caller should not
  //                 keep a reference to it. The caller can copy the data if
  //                 he/she wants to access the samples after the callback
  //                 returns. The callback is called in the current thread.
  // @param callback_arg The arg passed to callback, if callback is not NULL.
  GeneratedAudio Generate(const TtsArgs &args,
                          GeneratedAudioCallback callback = nullptr,
                          void *callback_arg = nullptr) const;

  // Return the sample rate of the generated audio
  int32_t SampleRate() const;

  // Number of supported speakers.
  // If it supports only a single speaker, then it return 0 or 1.
  int32_t NumSpeakers() const;

 private:
  std::unique_ptr<OfflineTtsImpl> impl_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_OFFLINE_TTS_H_
