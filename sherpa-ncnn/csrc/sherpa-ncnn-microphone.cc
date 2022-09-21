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

#include "portaudio.h"
#include "sherpa-ncnn/csrc/decode.h"
#include "sherpa-ncnn/csrc/features.h"
#include "sherpa-ncnn/csrc/lstm-model.h"
#include "sherpa-ncnn/csrc/microphone.h"
#include "sherpa-ncnn/csrc/symbol-table.h"

bool stop = false;

static int recordCallback(const void *input_buffer, void *outputBuffer,
                          unsigned long frames_per_buffer,
                          const PaStreamCallbackTimeInfo *timeInfo,
                          PaStreamCallbackFlags statusFlags, void *user_data) {
  auto feature_extractor =
      reinterpret_cast<sherpa_ncnn::FeatureExtractor *>(user_data);

  feature_extractor->AcceptWaveform(
      16000, reinterpret_cast<const float *>(input_buffer), frames_per_buffer);

  return stop ? paComplete : paContinue;
}
static void handler(int sig) {
  stop = true;
  fprintf(stderr, "\nexiting...\n");
};

int main(int32_t argc, char *argv[]) {
  if (argc != 8 && argc != 9) {
    const char *usage = R"usage(
Usage:
  ./bin/sherpa-ncnn \
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
  signal(SIGINT, handler);

  std::string tokens = argv[1];
  std::string encoder_param = argv[2];
  std::string encoder_bin = argv[3];
  std::string decoder_param = argv[4];
  std::string decoder_bin = argv[5];
  std::string joiner_param = argv[6];
  std::string joiner_bin = argv[7];

  int32_t num_threads = 4;
  if (argc == 9) {
    num_threads = atoi(argv[8]);
  }

  sherpa_ncnn::SymbolTable sym(tokens);
  fprintf(stderr, "Number of threads: %d\n", num_threads);

  sherpa_ncnn::LstmModel model(encoder_param, encoder_bin, decoder_param,
                               decoder_bin, joiner_param, joiner_bin,
                               num_threads);

  sherpa_ncnn::Microphone mic;

  sherpa_ncnn::FeatureExtractor feature_extractor;

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  fprintf(stderr, "num devices: %d\n", num_devices);

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
  float sample_rate = 16000;

  PaStream *stream;
  PaError err = Pa_OpenStream(&stream, &param, nullptr, /* &outputParameters, */
                              sample_rate,
                              0,         // frames per buffer
                              paClipOff, /* we won't output out of range samples
                                            so don't bother clipping them */
                              recordCallback,
                              &feature_extractor  // userdata
  );
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

  int32_t segment = 9;
  int32_t offset = 4;

  int32_t context_size = model.ContextSize();
  int32_t blank_id = model.BlankId();

  std::vector<int32_t> hyp(context_size, blank_id);

  ncnn::Mat decoder_input(context_size);
  for (int32_t i = 0; i != context_size; ++i) {
    static_cast<int32_t *>(decoder_input)[i] = blank_id;
  }

  ncnn::Mat decoder_out = model.RunDecoder(decoder_input);

  ncnn::Mat hx;
  ncnn::Mat cx;

  int32_t num_tokens = hyp.size();
  int32_t num_processed = 0;

  while (!stop) {
    while (feature_extractor.NumFramesReady() - num_processed >= segment) {
      ncnn::Mat features = feature_extractor.GetFrames(num_processed, segment);
      num_processed += offset;

      ncnn::Mat encoder_out = model.RunEncoder(features, &hx, &cx);

      GreedySearch(model, encoder_out, &decoder_out, &hyp);
    }

    if (hyp.size() != num_tokens) {
      num_tokens = hyp.size();
      std::string text;
      for (int32_t i = context_size; i != hyp.size(); ++i) {
        text += sym[hyp[i]];
      }
      fprintf(stderr, "%s\n", text.c_str());
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
