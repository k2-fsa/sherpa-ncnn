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

#include <algorithm>
#include <iostream>

#include "net.h"  // NOLINT
#include "sherpa-ncnn/csrc/recognizer.h"
#include "sherpa-ncnn/csrc/wave-reader.h"

int main(int argc, char *argv[]) {
  if (argc < 9 || argc > 11) {
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

You can download pre-trained models from the following repository:
https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05
)usage";
    std::cerr << usage << "\n";

    return 0;
  }
  sherpa_ncnn::ModelConfig model_conf;
  model_conf.tokens = argv[1];
  model_conf.encoder_param = argv[2];
  model_conf.encoder_bin = argv[3];
  model_conf.decoder_param = argv[4];
  model_conf.decoder_bin = argv[5];
  model_conf.joiner_param = argv[6];
  model_conf.joiner_bin = argv[7];
  int num_threads = 4;
  if (argc >= 10 && atoi(argv[9]) > 0) {
    num_threads = atoi(argv[9]);
  }
  model_conf.encoder_opt.num_threads = num_threads;
  model_conf.decoder_opt.num_threads = num_threads;
  model_conf.joiner_opt.num_threads = num_threads;

  const float expected_sampling_rate = 16000;
  sherpa_ncnn::DecoderConfig decoder_conf;
  if (argc == 11) {
    std::string method = argv[10];
    if (method.compare("greed_search") ||
        method.compare("modified_beam_search")) {
      decoder_conf.method = method;
    }
  }
  knf::FbankOptions fbank_opts;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.snip_edges = false;
  fbank_opts.frame_opts.samp_freq = expected_sampling_rate;
  fbank_opts.mel_opts.num_bins = 80;

  sherpa_ncnn::Recognizer recognizer(decoder_conf, model_conf, fbank_opts);

  std::string wav_filename = argv[8];

  std::cout << model_conf.ToString() << "\n";
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

  recognizer.AcceptWaveform(expected_sampling_rate, samples.data(),
                            samples.size());
  std::vector<float> tail_paddings(
      static_cast<int>(0.3 * expected_sampling_rate));
  recognizer.AcceptWaveform(expected_sampling_rate, tail_paddings.data(),
                            tail_paddings.size());

  recognizer.Decode();
  auto result = recognizer.GetResult();
  std::cout << "Recognition result for " << wav_filename << "\n"
            << result.text << "\n";

  return 0;
}
