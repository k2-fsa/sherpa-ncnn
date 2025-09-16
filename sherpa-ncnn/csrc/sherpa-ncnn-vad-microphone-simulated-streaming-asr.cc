// sherpa-ncnn/csrc/sherpa-ncnn-vad-microphone-simulated-streaming-asr.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>  // NOLINT
#include <mutex>   // NOLINT
#include <queue>
#include <string>
#include <vector>

#include "portaudio.h"  // NOLINT
#include "sherpa-ncnn/csrc/circular-buffer.h"
#include "sherpa-ncnn/csrc/microphone.h"
#include "sherpa-ncnn/csrc/offline-recognizer.h"
#include "sherpa-ncnn/csrc/resample.h"
#include "sherpa-ncnn/csrc/sherpa-display.h"
#include "sherpa-ncnn/csrc/voice-activity-detector.h"
#include "sherpa-ncnn/csrc/wave-writer.h"

std::queue<std::vector<float>> samples_queue;
std::condition_variable condition_variable;
std::mutex mutex;
bool stop = false;

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void * /*user_data*/) {
  std::lock_guard<std::mutex> lock(mutex);
  samples_queue.emplace(
      reinterpret_cast<const float *>(input_buffer),
      reinterpret_cast<const float *>(input_buffer) + frames_per_buffer);
  condition_variable.notify_one();

  return stop ? paComplete : paContinue;
}

static void Handler(int32_t /*sig*/) {
  stop = true;
  condition_variable.notify_one();
  fprintf(stdout, "\nCaught Ctrl + C. Exiting...\n");
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program shows how to use a streaming VAD with non-streaming ASR in
sherpa-ncnn for real-time speech recognition.

(1) SenseVoice

cd /path/to/sherpa-ncnn/build

wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/sherpa-ncnn-silero-vad.tar.bz2
tar xvf sherpa-ncnn-silero-vad.tar.bz2

wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/asr-models/sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

  ./bin/sherpa-ncnn-vad-microphone-simulated-streaming-asr \
    --silero-vad-model-dir=./sherpa-ncnn-silero-vad \
    --sense-voice-model-dir=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
    --tokens=./sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
    --num-threads=1

)usage";

  sherpa_ncnn::ParseOptions po(kUsageMessage);
  sherpa_ncnn::SileroVadModelConfig vad_config;

  sherpa_ncnn::OfflineRecognizerConfig asr_config;

  vad_config.Register(&po);
  asr_config.Register(&po);

  int32_t user_device_index = -1;  // -1 means to use default value
  int32_t user_sample_rate = -1;   // -1 means to use default value

  po.Register("mic-device-index", &user_device_index,
              "If provided, we use it to replace the default device index."
              "You can use sherpa-ncnn-pa-devs to list available devices");

  po.Register("mic-sample-rate", &user_sample_rate,
              "If provided, we use it to replace the default sample rate."
              "You can use sherpa-ncnn-pa-devs to list sample rate of "
              "available devices");

  if (argc == 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  po.Read(argc, argv);
  if (po.NumArgs() != 0) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stdout, "%s\n", vad_config.ToString().c_str());
  fprintf(stdout, "%s\n", asr_config.ToString().c_str());

  if (!vad_config.Validate()) {
    fprintf(stdout, "Errors in vad_config!\n");
    return -1;
  }

  if (!asr_config.Validate()) {
    fprintf(stdout, "Errors in asr_config!\n");
    return -1;
  }

  fprintf(stdout, "Creating recognizer ...\n");
  sherpa_ncnn::OfflineRecognizer recognizer(asr_config);
  fprintf(stdout, "Recognizer created!\n");

  sherpa_ncnn::Microphone mic;

  int32_t device_index = Pa_GetDefaultInputDevice();
  if (device_index == paNoDevice) {
    fprintf(stdout, "No default input device found\n");
    exit(EXIT_FAILURE);
  }

  if (user_device_index >= 0) {
    fprintf(stdout, "Use specified device: %d\n", user_device_index);
    device_index = user_device_index;
  } else {
    fprintf(stdout, "Use default device: %d\n", device_index);
  }

  mic.PrintDevices(device_index);

  float mic_sample_rate = 16000;
  if (user_sample_rate > 0) {
    fprintf(stdout, "Use sample rate %d for mic\n", user_sample_rate);
    mic_sample_rate = user_sample_rate;
  }

  if (!mic.OpenDevice(device_index, mic_sample_rate, 1, RecordCallback,
                      nullptr)) {
    fprintf(stdout, "Failed to open device %d\n", device_index);
    exit(EXIT_FAILURE);
  }

  float sample_rate = 16000;
  std::unique_ptr<sherpa_ncnn::LinearResample> resampler;
  if (mic_sample_rate != sample_rate) {
    float min_freq = std::min(mic_sample_rate, sample_rate);
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    resampler = std::make_unique<sherpa_ncnn::LinearResample>(
        mic_sample_rate, sample_rate, lowpass_cutoff, lowpass_filter_width);
  }

  auto vad = std::make_unique<sherpa_ncnn::VoiceActivityDetector>(vad_config);

  int32_t window_size = vad_config.window_size;

  int32_t offset = 0;
  bool speech_started = false;
  std::vector<float> buffer;

  auto started_time = std::chrono::steady_clock::now();
  sherpa_ncnn::SherpaDisplay display;

  fprintf(stdout, "Started. Please speak\n");
  std::vector<float> resampled;

  while (!stop) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      while (samples_queue.empty() && !stop) {
        condition_variable.wait(lock);
      }

      if (stop) {
        break;
      }

      const auto &s = samples_queue.front();
      if (!resampler) {
        buffer.insert(buffer.end(), s.begin(), s.end());
      } else {
        resampler->Resample(s.data(), s.size(), false, &resampled);
        buffer.insert(buffer.end(), resampled.begin(), resampled.end());
      }

      samples_queue.pop();
    }

    for (; offset + window_size < buffer.size(); offset += window_size) {
      vad->AcceptWaveform(buffer.data() + offset, window_size);
      if (!speech_started && vad->IsSpeechDetected()) {
        speech_started = true;
        started_time = std::chrono::steady_clock::now();
      }
    }

    if (!speech_started) {
      if (buffer.size() > 10 * window_size) {
        offset -= buffer.size() - 10 * window_size;
        buffer = {buffer.end() - 10 * window_size, buffer.end()};
      }
    }

    auto current_time = std::chrono::steady_clock::now();
    const float elapsed_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time -
                                                              started_time)
            .count() /
        1000.;

    if (speech_started && elapsed_seconds > 0.2) {
      auto s = recognizer.CreateStream();
      s->AcceptWaveform(sample_rate, buffer.data(), buffer.size());
      recognizer.DecodeStream(s.get());
      const auto &result = s->GetResult();
      display.UpdateText(result.text);
      display.Display();

      started_time = std::chrono::steady_clock::now();
    }

    while (!vad->Empty()) {
      // when stopping speak, this while loop is executed

      vad->Pop();

      display.FinalizeCurrentSentence();
      display.Display();

      buffer.clear();
      offset = 0;
      speech_started = false;
    }
  }

  return 0;
}
