// sherpa-ncnn/csrc/offline-tts.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-tts.h"

#include <cmath>
#include <string>
#include <utility>

#include "sherpa-ncnn/csrc/file-utils.h"
#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/offline-tts-impl.h"
#include "sherpa-ncnn/csrc/text-utils.h"

namespace sherpa_ncnn {

struct SilenceInterval {
  int32_t start;
  int32_t end;
};

GeneratedAudio GeneratedAudio::ScaleSilence(float scale) const {
  if (scale == 1) {
    return *this;
  }
  // if the interval is larger than 0.2 second, then we assume it is a pause
  int32_t threshold = static_cast<int32_t>(sample_rate * 0.2);

  std::vector<SilenceInterval> intervals;
  int32_t num_samples = static_cast<int32_t>(samples.size());

  int32_t last = -1;
  int32_t i;
  for (i = 0; i != num_samples; ++i) {
    if (fabs(samples[i]) <= 0.01) {
      if (last == -1) {
        last = i;
      }
      continue;
    }

    if (last != -1 && i - last < threshold) {
      last = -1;
      continue;
    }

    if (last != -1) {
      intervals.push_back({last, i});
      last = -1;
    }
  }

  if (last != -1 && num_samples - last > threshold) {
    intervals.push_back({last, num_samples});
  }

  if (intervals.empty()) {
    return *this;
  }

  GeneratedAudio ans;
  ans.sample_rate = sample_rate;
  ans.samples.reserve(samples.size());

  i = 0;
  for (const auto &interval : intervals) {
    ans.samples.insert(ans.samples.end(), samples.begin() + i,
                       samples.begin() + interval.start);
    i = interval.end;
    int32_t n = static_cast<int32_t>((interval.end - interval.start) * scale);

    ans.samples.insert(ans.samples.end(), samples.begin() + interval.start,
                       samples.begin() + interval.start + n);
  }

  if (i < num_samples) {
    ans.samples.insert(ans.samples.end(), samples.begin() + i, samples.end());
  }

  return ans;
}

void OfflineTtsConfig::Register(ParseOptions *po) {
  model.Register(po);

  po->Register("tts-rule-fsts", &rule_fsts,
               "If not empty, it contains a list of rule FST filenames."
               "Multiple filenames are separated by a comma and they are "
               "applied from left to right. An example value: "
               "rule1.fst,rule2.fst,rule3.fst");

  po->Register("tts-rule-fars", &rule_fars,
               "If not empty, it contains a list of rule FST archive filenames."
               "Multiple filenames are separated by a comma and they are "
               "applied from left to right. An example value: "
               "rule1.far,rule2.far,rule3.far. Note that an *.far can contain "
               "multiple *.fst files");

  po->Register(
      "tts-max-num-sentences", &max_num_sentences,
      "Maximum number of sentences that we process at a time. "
      "This is to avoid OOM for very long input text. "
      "If you set it to -1, then we process all sentences in a single batch.");

  po->Register(
      "tts-max-tokens-per-sentence", &max_tokens_per_sentence,
      "If positive, we limit the number of tokens per sentence to this value");

  po->Register("tts-silence-scale", &silence_scale,
               "Duration of the pause is scaled by this number. So a smaller "
               "value leads to a shorter pause.");
}

bool OfflineTtsConfig::Validate() const {
  if (!rule_fsts.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fsts, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_NCNN_LOGE("Rule fst '%s' does not exist. ", f.c_str());
        return false;
      }
    }
  }

  if (!rule_fars.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(rule_fars, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_NCNN_LOGE("Rule far '%s' does not exist. ", f.c_str());
        return false;
      }
    }
  }

  if (silence_scale < 0.001) {
    SHERPA_NCNN_LOGE("--tts-silence-scale '%.3f' is too small", silence_scale);
    return false;
  }

  return model.Validate();
}

std::string OfflineTtsConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsConfig(";
  os << "model=" << model.ToString() << ", ";
  os << "rule_fsts=\"" << rule_fsts << "\", ";
  os << "rule_fars=\"" << rule_fars << "\", ";
  os << "max_num_sentences=" << max_num_sentences << ", ";
  os << "silence_scale=" << silence_scale << ")";

  return os.str();
}

OfflineTts::OfflineTts(const OfflineTtsConfig &config)
    : impl_(OfflineTtsImpl::Create(config)) {}

OfflineTts::~OfflineTts() = default;

GeneratedAudio OfflineTts::Generate(
    const TtsArgs &args, GeneratedAudioCallback callback /*= nullptr*/,
    void *callback_arg /*= nullptr*/) const {
#if !defined(_WIN32)
  return impl_->Generate(args, std::move(callback), callback_arg);
#else
  if (IsUtf8(args.text)) {
    return impl_->Generate(args, std::move(callback), callback_arg);
  } else if (IsGB2312(args.text)) {
    auto utf8_text = Gb2312ToUtf8(args.text);
    static bool printed = false;
    if (!printed) {
      SHERPA_NCNN_LOGE(
          "Detected GB2312 encoded string! Converting it to UTF8.");
      printed = true;
    }
    TtsArgs utf8_args = args;
    utf8_args.text = utf8_text;
    return impl_->Generate(utf8_args, std::move(callback), callback_arg);
  } else {
    SHERPA_NCNN_LOGE(
        "Non UTF8 encoded string is received. You would not get expected "
        "results!");
    return impl_->Generate(args, std::move(callback), callback_arg);
  }
#endif
}

int32_t OfflineTts::SampleRate() const { return impl_->SampleRate(); }

int32_t OfflineTts::NumSpeakers() const { return impl_->NumSpeakers(); }

}  // namespace sherpa_ncnn
