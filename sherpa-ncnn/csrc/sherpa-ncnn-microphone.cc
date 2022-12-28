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
#include "sherpa-ncnn/csrc/decode.h"
#include "sherpa-ncnn/csrc/features.h"
#include "sherpa-ncnn/csrc/microphone.h"
#include "sherpa-ncnn/csrc/model.h"
#include "sherpa-ncnn/csrc/symbol-table.h"

bool stop = false;

static int RecordCallback(const void *input_buffer, void * /*output_buffer*/,
                          unsigned long frames_per_buffer,  // NOLINT
                          const PaStreamCallbackTimeInfo * /*time_info*/,
                          PaStreamCallbackFlags /*status_flags*/,
                          void *user_data) {
  auto feature_extractor =
      reinterpret_cast<sherpa_ncnn::FeatureExtractor *>(user_data);

  feature_extractor->AcceptWaveform(
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

  sherpa_ncnn::ModelConfig config;

  config.tokens = argv[1];
  config.encoder_param = argv[2];
  config.encoder_bin = argv[3];
  config.decoder_param = argv[4];
  config.decoder_bin = argv[5];
  config.joiner_param = argv[6];
  config.joiner_bin = argv[7];

  int32_t num_threads = 4;
  if (argc == 9) {
    num_threads = atoi(argv[8]);
  }

  config.encoder_opt.num_threads = num_threads;
  config.decoder_opt.num_threads = num_threads;
  config.joiner_opt.num_threads = num_threads;

  sherpa_ncnn::SymbolTable sym(config.tokens);
  fprintf(stderr, "%s\n", config.ToString().c_str());

  auto model = sherpa_ncnn::Model::Create(config);
  if (!model) {
    fprintf(stderr, "Failed to create a model\n");
    exit(EXIT_FAILURE);
  }

  float sample_rate = 16000;
  sherpa_ncnn::Microphone mic;

  knf::FbankOptions fbank_opts;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.snip_edges = false;
  fbank_opts.frame_opts.samp_freq = sample_rate;
  fbank_opts.mel_opts.num_bins = 80;

  sherpa_ncnn::FeatureExtractor feature_extractor(fbank_opts);

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

  PaStream *stream;
  PaError err =
      Pa_OpenStream(&stream, &param, nullptr, /* &outputParameters, */
                    sample_rate,
                    0,          // frames per buffer
                    paClipOff,  // we won't output out of range samples
                                // so don't bother clipping them
                    RecordCallback, &feature_extractor);
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

  int32_t segment = model->Segment();
  int32_t offset = model->Offset();

  int32_t context_size = model->ContextSize();
  int32_t blank_id = model->BlankId();

  std::vector<int32_t> hyp(context_size, blank_id);

  ncnn::Mat decoder_input(context_size);
  for (int32_t i = 0; i != context_size; ++i) {
    static_cast<int32_t *>(decoder_input)[i] = blank_id;
  }

  ncnn::Mat decoder_out = model->RunDecoder(decoder_input);

  ncnn::Mat hx;
  ncnn::Mat cx;

  int32_t num_tokens = hyp.size();
  int32_t num_processed = 0;

  std::vector<ncnn::Mat> states;
  ncnn::Mat encoder_out;

  while (!stop) {
    while (feature_extractor.NumFramesReady() - num_processed >= segment) {
      ncnn::Mat features = feature_extractor.GetFrames(num_processed, segment);
      num_processed += offset;

      std::tie(encoder_out, states) = model->RunEncoder(features, states);

      GreedySearch(model.get(), encoder_out, &decoder_out, &hyp);
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
