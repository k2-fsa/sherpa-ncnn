/**
 * Copyright (c) 2026 Yantai Xiaoyingtao Technology. (authors: Seven Du)
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

#include <chrono>  // NOLINT
#include <cstdio>

#include "sherpa-ncnn/c-api/c-api.h"
#include "sherpa-ncnn/csrc/offline-tts.h"

struct SherpaNcnnTts {
  std::unique_ptr<sherpa_ncnn::OfflineTts> tts;
};

struct SherpaNcnnTtsAudio {
  float elapsed_seconds;
  uint32_t sample_rate;
  uint32_t nsamples;
  float duration;
  float rtf;
  std::unique_ptr<sherpa_ncnn::GeneratedAudio> audio;
};

static int32_t DefaultAudioCallback(const float* /*samples*/,
                                    int32_t num_samples, int32_t processed,
                                    int32_t total, void* arg) {
  float progress = static_cast<float>(processed) / total;
  printf("Progress=%.3f%%\n", progress * 100);

  return 1;
}

SherpaNcnnTts* SherpaNcnnCreateTts(const SherpaNcnnTtsConfig* in_config) {
  sherpa_ncnn::OfflineTtsConfig config;

  if (!in_config || !in_config->model_dir) {
    fprintf(stderr, "Invalid config!\n");
    return nullptr;
  }

  config.model.debug = in_config->debug;
  config.model.vits.model_dir = std::string(in_config->model_dir);
  if (in_config->n_threads > 0) config.model.num_threads = in_config->n_threads;
  // config.max_tokens_per_sentence = -1;

  if (config.model.debug) {
    fprintf(stderr, "%s\n", config.model.ToString().c_str());
  }

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return nullptr;
  }

  auto tts = std::make_unique<sherpa_ncnn::OfflineTts>(config);
  auto ntts = new SherpaNcnnTts;
  ntts->tts = std::move(tts);
  return ntts;
}

SherpaNcnnTtsAudio* SherpaNcnnTtsGenerate(SherpaNcnnTts* ntts, const char* text,
                                          const SherpaNcnnTtsConfig* config) {
    return SherpaNcnnTtsGenerateEx(ntts, text, config, DefaultAudioCallback, NULL);
}

SherpaNcnnTtsAudio* SherpaNcnnTtsGenerateEx(SherpaNcnnTts* ntts, const char* text,
                                          const SherpaNcnnTtsConfig* config,
                                          SherpaNcnnTtsAudioCallback callback,
                                          void* user_data) {
  if (!ntts || !text || !config) {
    fprintf(stderr, "Invalid arguments!\n");
    return nullptr;
  }

  const auto begin = std::chrono::steady_clock::now();
  sherpa_ncnn::TtsArgs args;
  args.text = text;
  args.sid = config->sid;
  args.speed = config->speed > 0.1 ? config->speed : 1.0;
  auto audio = ntts->tts->Generate(args, callback, user_data);
  const auto end = std::chrono::steady_clock::now();

  if (audio.samples.empty()) {
    fprintf(
        stderr,
        "Error in generating audio. Please read previous error messages.\n");
    return nullptr;
  }

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = audio.samples.size() / static_cast<float>(audio.sample_rate);

  float rtf = elapsed_seconds / duration;

  auto a = std::make_unique<SherpaNcnnTtsAudio>();
  a->elapsed_seconds = elapsed_seconds;
  a->duration = duration;
  a->sample_rate = audio.sample_rate;
  a->nsamples = audio.samples.size();
  a->rtf = rtf;
  a->audio = std::make_unique<sherpa_ncnn::GeneratedAudio>(std::move(audio));
  return a.release();
}

void SherpaNcnnTtsDestroy(SherpaNcnnTts** ntts) {
  if (!ntts || !*ntts) return;
  delete *ntts;
  *ntts = nullptr;
}

void SherpaNcnnTtsAudioDestroy(SherpaNcnnTtsAudio** audio) {
  if (!audio || !*audio) return;
  delete *audio;
  *audio = nullptr;
}

int32_t SherpaNcnnTtsAudioGetSampleRate(const SherpaNcnnTtsAudio* audio) {
  return audio->sample_rate;
}

float* SherpaNcnnTtsAudioGetSamplePtr(const SherpaNcnnTtsAudio* audio) {
  return audio->audio->samples.data();
}

uint32_t SherpaNcnnTtsAudioGetSampleCount(const SherpaNcnnTtsAudio* audio) {
  return audio->nsamples;
}
float SherpaNcnnTtsAudioGetDuration(const SherpaNcnnTtsAudio* audio) {
  return audio->duration;
}

float SherpaNcnnTtsAudioGetRtf(const SherpaNcnnTtsAudio* audio) { return audio->rtf; }

float SherpaNcnnTtsAudioGetElapsedSeconds(const SherpaNcnnTtsAudio* audio) {
  return audio->elapsed_seconds;
}
