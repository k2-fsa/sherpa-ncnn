/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include "portaudio.h"  // NOLINT
#include "sherpa-ncnn/csrc/decoder.h"
#include "sherpa-ncnn/csrc/microphone.h"

bool stop = false;

static int RecordCallback(const void *input_buffer, void * /*output_buffer*/,
                          unsigned long frames_per_buffer,  // NOLINT
                          const PaStreamCallbackTimeInfo * /*time_info*/,
                          PaStreamCallbackFlags /*status_flags*/,
                          void *user_data) {
  auto recognizer =
      reinterpret_cast<sherpa_ncnn::Recognizer *>(user_data);

  recognizer->AcceptWaveform(
      16000, reinterpret_cast<const float *>(input_buffer), frames_per_buffer);

  return stop ? paComplete : paContinue;
}

static void Handler(int sig) {
  stop = true;
  fprintf(stderr, "\nexiting...\n");
};

int main(int32_t argc, char *argv[]) {
  if (argc != 8 && argc != 9) {
    const char *usage = R"usage(
Usage:
  ./bin/sherpa-ncnn-microphone \
    /path/to/tokens.txt \
    /path/to/encoder.ncnn.param \
    /path/to/encoder.ncnn.bin \
    /path/to/decoder.ncnn.param \
    /path/to/decoder.ncnn.bin \
    /path/to/joiner.ncnn.param \
    /path/to/joiner.ncnn.bin \
    [num_threads]

You can download pre-trained models from the following repository:
https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05
)usage";
    fprintf(stderr, "%s\n", usage);
    fprintf(stderr, "argc, %d\n", argc);

    return 0;
  }
  signal(SIGINT, Handler);

  sherpa_ncnn::ModelConfig model_conf;
  model_conf.tokens_fn = argv[1];
  model_conf.encoder_param = argv[2];
  model_conf.encoder_bin = argv[3];
  model_conf.decoder_param = argv[4];
  model_conf.decoder_bin = argv[5];
  model_conf.joiner_param = argv[6];
  model_conf.joiner_bin = argv[7];
  model_conf.num_threads = 4;
  if (argc == 9) {
    model_conf.num_threads = atoi(argv[8]);
  }
  sherpa_ncnn::DecoderConfig decoder_conf;


  sherpa_ncnn::Recognizer recognizer(decoder_conf, model_conf);

  sherpa_ncnn::Microphone mic;

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  fprintf(stderr, "Num devices: %d\n", num_devices);

  PaStreamParameters param;

  param.device = Pa_GetDefaultInputDevice();
  if (param.device == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Use default device: %d\n", param.device);

  const PaDeviceInfo *info = Pa_GetDeviceInfo(param.device);
  fprintf(stderr, "  Name: %s\n", info->name);
  fprintf(stderr, "  Max input channels: %d\n", info->maxInputChannels);

  param.channelCount = 1;
  param.sampleFormat = paFloat32;

  param.suggestedLatency = info->defaultLowInputLatency;
  param.hostApiSpecificStreamInfo = nullptr;
  const float sample_rate = 16000;

  PaStream *stream;
  PaError err =
      Pa_OpenStream(&stream, &param, nullptr, /* &outputParameters, */
                    sample_rate,
                    0,          // frames per buffer
                    paClipOff,  // we won't output out of range samples
                                // so don't bother clipping them
                    RecordCallback, &recognizer);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  err = Pa_StartStream(stream);
  fprintf(stderr, "Started\n");

  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  int num_tokens = 0;
  while (!stop) {
    recognizer.Decode();
    auto result = recognizer.GetResult();
    if (result.text.size() != num_tokens) {
      fprintf(stderr, "%s\n", result.text.c_str());
    }

    Pa_Sleep(20);  // sleep for 20ms
  }

  err = Pa_CloseStream(stream);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  return 0;
}
