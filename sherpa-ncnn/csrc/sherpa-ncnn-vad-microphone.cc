// sherpa-ncnn/csrc/sherpa-ncnn-vad-microphone.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <mutex>  // NOLINT

#include "portaudio.h"  // NOLINT
#include "sherpa-ncnn/csrc/circular-buffer.h"
#include "sherpa-ncnn/csrc/microphone.h"
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

int32_t main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
This file shows how to use silero vad with a microphone.

===========Usage============:

0. Build sherpa-ncnn
--------------------

mkdir -p $HOME/open-source
cd $HOME/open-source
git clone https://github.com/k2-fsa/sherpa-ncnn
cd sherpa-ncnn
mkdir build
cd build
cmake ..
make -j3

1. Download the vad model
-------------------------

cd $HOME/open-source/sherpa-ncnn/build
wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/sherpa-ncnn-silero-vad.tar.bz2
tar xvf sherpa-ncnn-silero-vad.tar.bz2

2. Run it!
----------

cd $HOME/open-source/sherpa-ncnn/build

./bin/sherpa-ncnn-vad-microphone --silero-vad-model-dir=sherpa-ncnn-silero-vad
)usage";

  sherpa_ncnn::ParseOptions po(kUsageMessage);
  sherpa_ncnn::SileroVadModelConfig config;

  config.Register(&po);

  int32_t user_device_index = -1;  // -1 means to use default value
  int32_t user_sample_rate = -1;   // -1 means to use default value

  po.Register("mic-device-index", &user_device_index,
              "If provided, we use it to replace the default device index."
              "You can use sherpa-ncnn-pa-devs to list available devices");

  po.Register("mic-sample-rate", &user_sample_rate,
              "If provided, we use it to replace the default sample rate."
              "You can use sherpa-ncnn-pa-devs to list sample rate of "
              "available devices");

  po.Read(argc, argv);

  if (argc == 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (po.NumArgs() != 0) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stdout, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stdout, "Errors in config!\n");
    return -1;
  }

  sherpa_ncnn::Microphone mic;

  int32_t device_index = Pa_GetDefaultInputDevice();
  if (device_index == paNoDevice) {
    fprintf(stdout, "No default input device found\n");
    fprintf(stdout, "If you are using Linux, please switch to \n");
    fprintf(stdout, " ./bin/sherpa-ncnn-vad-alsa \n");
    exit(EXIT_FAILURE);
  }

  if (user_device_index >= 0) {
    fprintf(stdout, "Use specified device: %d\n", user_device_index);
    device_index = user_device_index;
  } else {
    fprintf(stdout, "Use default device: %d\n", device_index);
  }

  float mic_sample_rate = 16000;
  if (user_sample_rate > 0) {
    fprintf(stdout, "Use sample rate %d for mic\n", user_sample_rate);
    mic_sample_rate = user_sample_rate;
  }

  if (!mic.OpenDevice(device_index, mic_sample_rate, 1, RecordCallback,
                      nullptr)) {
    fprintf(stdout, "Failed to open microphone device %d\n", device_index);
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

  auto vad = std::make_unique<sherpa_ncnn::VoiceActivityDetector>(config);

  int32_t window_size = config.window_size;
  bool printed = false;

  int32_t k = 0;
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

        if (vad->IsSpeechDetected() && !printed) {
          printed = true;
          fprintf(stdout, "\nDetected speech!\n");
        }
        if (!vad->IsSpeechDetected()) {
          printed = false;
        }

        while (!vad->Empty()) {
          const auto &segment = vad->Front();
          float duration = segment.samples.size() / sample_rate;
          fprintf(stdout, "Duration: %.3f seconds\n", duration);

          char filename[128];
          snprintf(filename, sizeof(filename), "seg-%d-%.3fs.wav", k, duration);
          k += 1;
          sherpa_ncnn::WriteWave(filename, sample_rate, segment.samples.data(),
                                 segment.samples.size());
          fprintf(stdout, "Saved to %s\n", filename);
          fprintf(stdout, "----------\n");

          vad->Pop();
        }
      }
    }
    Pa_Sleep(100);  // sleep for 100ms
  }

  return 0;
}
