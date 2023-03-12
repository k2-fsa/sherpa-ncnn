/**
 * Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include <algorithm>
#include <cctype>  // std::tolower
#include <cstdint>

#include "sherpa-ncnn/csrc/alsa.h"
#include "sherpa-ncnn/csrc/display.h"
#include "sherpa-ncnn/csrc/recognizer.h"

bool stop = false;

static void Handler(int sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
};

int main(int32_t argc, char *argv[]) {
  if (argc < 9 || argc > 11) {
    const char *usage = R"usage(
Usage:
  ./bin/sherpa-ncnn-alsa \
    /path/to/tokens.txt \
    /path/to/encoder.ncnn.param \
    /path/to/encoder.ncnn.bin \
    /path/to/decoder.ncnn.param \
    /path/to/decoder.ncnn.bin \
    /path/to/joiner.ncnn.param \
    /path/to/joiner.ncnn.bin \
    device_name \
    [num_threads] [decode_method, can be greedy_search/modified_beam_search]

Please refer to
https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
for a list of pre-trained models to download.

The device name specifies which microphone to use in case there are several
on you system. You can use

  arecord -l

to find all available microphones on your computer. For instance, if it outputs

**** List of CAPTURE Hardware Devices ****
card 3: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

and if you want to select card 3 and the device 0 on that card, please use:

  hw:3,0

as the device_name.
)usage";

    fprintf(stderr, "%s\n", usage);
    fprintf(stderr, "argc, %d\n", argc);

    return 0;
  }

  signal(SIGINT, Handler);

  sherpa_ncnn::RecognizerConfig config;
  config.model_config.tokens = argv[1];
  config.model_config.encoder_param = argv[2];
  config.model_config.encoder_bin = argv[3];
  config.model_config.decoder_param = argv[4];
  config.model_config.decoder_bin = argv[5];
  config.model_config.joiner_param = argv[6];
  config.model_config.joiner_bin = argv[7];

  const char *device_name = argv[8];

  int num_threads = 4;
  if (argc >= 10 && atoi(argv[9]) > 0) {
    num_threads = atoi(argv[9]);
  }

  config.model_config.encoder_opt.num_threads = num_threads;
  config.model_config.decoder_opt.num_threads = num_threads;
  config.model_config.joiner_opt.num_threads = num_threads;

  sherpa_ncnn::DecoderConfig decoder_conf;
  if (argc == 11) {
    std::string method = argv[10];
    if (method == "greedy_search" || method == "modified_beam_search") {
      decoder_conf.method = method;
    }
  }

  int32_t expected_sampling_rate = 16000;

  config.enable_endpoint = true;

  config.endpoint_config.rule1.min_trailing_silence = 2.4;
  config.endpoint_config.rule2.min_trailing_silence = 1.2;
  config.endpoint_config.rule3.min_utterance_length = 300;

  config.feat_config.sampling_rate = expected_sampling_rate;
  config.feat_config.feature_dim = 80;

  fprintf(stderr, "%s\n", config.ToString().c_str());

  sherpa_ncnn::Recognizer recognizer(config);

  sherpa_ncnn::Alsa alsa(device_name);
  fprintf(stderr, "Use recording device: %s\n", device_name);

  if (alsa.GetExpectedSampleRate() != expected_sampling_rate) {
    fprintf(stderr, "sample rate: %d != %d\n", alsa.GetExpectedSampleRate(),
            expected_sampling_rate);
    exit(-1);
  }

  auto s = recognizer.CreateStream();

  int32_t chunk = 0.1 * alsa.GetActualSampleRate();

  std::string last_text;
  int32_t segment_index = 0;
  sherpa_ncnn::Display display;
  while (!stop) {
    const std::vector<float> samples = alsa.Read(chunk);

    s->AcceptWaveform(expected_sampling_rate, samples.data(), samples.size());
    while (recognizer.IsReady(s.get())) {
      recognizer.DecodeStream(s.get());
    }

    bool is_endpoint = recognizer.IsEndpoint(s.get());
    auto text = recognizer.GetResult(s.get()).text;

    if (!text.empty() && last_text != text) {
      last_text = text;

      std::transform(text.begin(), text.end(), text.begin(),
                     [](auto c) { return std::tolower(c); });

      display.Print(segment_index, text);
    }

    if (is_endpoint) {
      if (!text.empty()) {
        ++segment_index;
      }

      recognizer.Reset(s.get());
    }
  }

  return 0;
}
