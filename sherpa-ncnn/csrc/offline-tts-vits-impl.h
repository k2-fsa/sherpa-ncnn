// sherpa-ncnn/csrc/offline-tts-vits-impl.h
//
// Copyright (c)  2023-2025  Xiaomi Corporation
#ifndef SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_IMPL_H_
#define SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_IMPL_H_

#include <stdlib.h>

#include <memory>
#include <string>
#include <strstream>
#include <utility>
#include <vector>

#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/offline-tts-vits-model.h"
#if 0
#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-ncnn/csrc/file-utils.h"
#include "sherpa-ncnn/csrc/lexicon.h"
#include "sherpa-ncnn/csrc/melo-tts-lexicon.h"
#include "sherpa-ncnn/csrc/offline-tts-character-frontend.h"
#include "sherpa-ncnn/csrc/offline-tts-frontend.h"
#include "sherpa-ncnn/csrc/offline-tts-impl.h"
#include "sherpa-ncnn/csrc/piper-phonemize-lexicon.h"
#include "sherpa-ncnn/csrc/text-utils.h"
#endif

namespace sherpa_ncnn {

class OfflineTtsVitsImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsVitsImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsVitsModel>(config.model)) {}

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

    const auto &meta_data = model_->GetMetaData();
    int32_t num_speakers = meta_data.num_speakers;
    SHERPA_NCNN_LOGE("num_speakers: %d", num_speakers);

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

      ncnn::Mat o =
          Process(tokens, args.noise_scale_w, args.noise_scale, args.speed);

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
  ncnn::Mat Process(const std::vector<int32_t> &tokens, float noise_scale_w,
                    float noise_scale, float speed) const {
    ncnn::Mat sequence(tokens.size(), 1);
    std::copy(tokens.begin(), tokens.end(), static_cast<int32_t *>(sequence));

    auto encoder_out = model_->RunEncoder(sequence);
    sequence.release();

    ncnn::Mat noise(encoder_out[0].w, 2);
    for (int32_t i = 0; i != noise.w * noise.h; ++i) {
      noise[i] = rand() / (float)RAND_MAX * noise_scale_w;
    }

    ncnn::Mat logw = model_->RunDurationPredictor(encoder_out[0], noise);

    noise.release();
    encoder_out[0].release();

    ncnn::Mat z_p = model_->PathAttention(logw, encoder_out[1], encoder_out[2],
                                          noise_scale, speed);
    encoder_out.clear();
    logw.release();

    ncnn::Mat z = model_->RunFlow(z_p);
    z_p.release();

    ncnn::Mat o = model_->RunDecoder(z);
    z.release();
    return o;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsVitsModel> model_;
};

}  // namespace sherpa_ncnn
#endif  // SHERPA_NCNN_CSRC_OFFLINE_TTS_VITS_IMPL_H_
