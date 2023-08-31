/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 * Copyright (c)  2022                     (Pingfeng Luo)
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

#include <stdio.h>

#include <algorithm>
#include <chrono>  // NOLINT
#include <fstream>
#include <iostream>

#include "net.h"  // NOLINT
#include "sherpa-ncnn/csrc/recognizer.h"
#include "sherpa-ncnn/csrc/wave-reader.h"

int32_t main(int32_t argc, char *argv[]) {
  if (argc < 9 || argc > 13) {
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
    /path/to/foo.wav [num_threads] [decode_method, can be greedy_search/modified_beam_search]

Please refer to
https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";
    std::cerr << usage << "\n";

    return 0;
  }
  sherpa_ncnn::RecognizerConfig config;
  config.model_config.tokens = argv[1];
  config.model_config.encoder_param = argv[2];
  config.model_config.encoder_bin = argv[3];
  config.model_config.decoder_param = argv[4];
  config.model_config.decoder_bin = argv[5];
  config.model_config.joiner_param = argv[6];
  config.model_config.joiner_bin = argv[7];
  int32_t num_threads = 4;
  if (argc >= 10 && atoi(argv[9]) > 0) {
    num_threads = atoi(argv[9]);
  }
  config.model_config.encoder_opt.num_threads = num_threads;
  config.model_config.decoder_opt.num_threads = num_threads;
  config.model_config.joiner_opt.num_threads = num_threads;

  float expected_sampling_rate = 16000;
  if (argc >= 11) {
    std::string method = argv[10];
    if (method == "greedy_search" || method == "modified_beam_search") {
      config.decoder_config.method = method;
    }
  }

  if (argc >= 12) {
    config.hotwords_file = argv[11];
  }

  if (argc == 13) {
    config.hotwords_score = atof(argv[12]);
  }

  config.feat_config.sampling_rate = expected_sampling_rate;
  config.feat_config.feature_dim = 80;

  std::cout << config.ToString() << "\n";

  sherpa_ncnn::Recognizer recognizer(config);

  std::string wav_filename = argv[8];

  bool is_ok = false;
  std::vector<float> samples =
      sherpa_ncnn::ReadWave(wav_filename, expected_sampling_rate, &is_ok);
  if (!is_ok) {
    fprintf(stderr, "Failed to read %s\n", wav_filename.c_str());
    exit(-1);
  }

  const float duration = samples.size() / expected_sampling_rate;
  std::cout << "wav filename: " << wav_filename << "\n";
  std::cout << "wav duration (s): " << duration << "\n";

  auto begin = std::chrono::steady_clock::now();
  std::cout << "Started!\n";
  auto stream = recognizer.CreateStream();
  stream->AcceptWaveform(expected_sampling_rate, samples.data(),
                         samples.size());
  std::vector<float> tail_paddings(
      static_cast<int>(0.3 * expected_sampling_rate));
  stream->AcceptWaveform(expected_sampling_rate, tail_paddings.data(),
                         tail_paddings.size());

  while (recognizer.IsReady(stream.get())) {
    recognizer.DecodeStream(stream.get());
  }
  auto result = recognizer.GetResult(stream.get());
  std::cout << "Done!\n";

  std::cout << "Recognition result for " << wav_filename << "\n"
            << result.ToString();

  auto end = std::chrono::steady_clock::now();
  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);

  return 0;
}
