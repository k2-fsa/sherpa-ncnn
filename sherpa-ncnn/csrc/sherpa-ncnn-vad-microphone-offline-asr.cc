// sherpa-ncnn/csrc/sherpa-ncnn-vad-microphone-offline-asr.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>  // NOLINT
#include <iomanip>
#include <mutex>  // NOLINT
#include <sstream>
#include <string>
#include <vector>

#include "portaudio.h"  // NOLINT
#include "sherpa-ncnn/csrc/circular-buffer.h"
#include "sherpa-ncnn/csrc/microphone.h"
#include "sherpa-ncnn/csrc/offline-recognizer.h"
#include "sherpa-ncnn/csrc/resample.h"
#include "sherpa-ncnn/csrc/voice-activity-detector.h"
#include "sherpa-ncnn/csrc/wave-writer.h"

bool stop = false;
std::mutex mutex;
sherpa_ncnn::CircularBuffer buffer(16000 * 60);

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void * /*user_data*/) {
  std::lock_guard<std::mutex> lock(mutex);
  buffer.Push(reinterpret_cast<const float *>(input_buffer), frames_per_buffer);

  return stop ? paComplete : paContinue;
}

static void Handler(int32_t /*sig*/) {
  stop = true;
  fprintf(stdout, "\nCaught Ctrl + C. Exiting...\n");
}

static std::string GetCurrentDatetimeAsString() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time = *std::localtime(&now_time);

  std::stringstream ss;
  ss << std::put_time(&local_time, "%Y-%m-%d-%H-%M-%S");

  return ss.str();
}

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This program shows how to use a streaming VAD with non-streaming ASR in
sherpa-ncnn.

(1) SenseVoice

cd /path/to/sherpa-ncnn/build

wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/sherpa-ncnn-silero-vad.tar.bz2
tar xvf sherpa-ncnn-silero-vad.tar.bz2

wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/asr-models/sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-ncnn-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

  ./bin/sherpa-ncnn-vad-microphone-offline-asr \
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

  if (argc == 0) {
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

  fprintf(stdout, "Started. Please speak\n");

  int32_t window_size = vad_config.window_size;
  int32_t index = 0;

  while (!stop) {
    {
      std::lock_guard<std::mutex> lock(mutex);

      while (buffer.Size() >= window_size) {
        std::vector<float> samples = buffer.Get(buffer.Head(), window_size);
        buffer.Pop(window_size);

        if (resampler) {
          std::vector<float> tmp;
          resampler->Resample(samples.data(), samples.size(), true, &tmp);
          samples = std::move(tmp);
        }

        vad->AcceptWaveform(samples.data(), samples.size());
      }
    }

    while (!vad->Empty()) {
      const auto &segment = vad->Front();
      auto s = recognizer.CreateStream();
      s->AcceptWaveform(sample_rate, segment.samples.data(),
                        segment.samples.size());
      recognizer.DecodeStream(s.get());
      const auto &result = s->GetResult();
      if (!result.text.empty()) {
        fprintf(stdout, "%2d: %s\n", index, result.text.c_str());
        std::string filename = GetCurrentDatetimeAsString() + "-" +
                               std::to_string(index) + "-" + result.text +
                               ".wav";
        sherpa_ncnn::WriteWave(filename, sample_rate, segment.samples.data(),
                               segment.samples.size());
        ++index;
      }
      vad->Pop();
    }

    Pa_Sleep(100);  // sleep for 100ms
  }

  return 0;
}
