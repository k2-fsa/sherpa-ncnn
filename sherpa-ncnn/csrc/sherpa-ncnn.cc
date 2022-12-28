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

#include <algorithm>
#include <iostream>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "net.h"  // NOLINT
#include "sherpa-ncnn/csrc/decode.h"
#include "sherpa-ncnn/csrc/features.h"
#include "sherpa-ncnn/csrc/model.h"
#include "sherpa-ncnn/csrc/symbol-table.h"
#include "sherpa-ncnn/csrc/wave-reader.h"

int main(int argc, char *argv[]) {
  if (argc < 9 || argc > 10) {
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
    /path/to/foo.wav [num_threads]

You can download pre-trained models from the following repository:
https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05
)usage";
    std::cerr << usage << "\n";

    return 0;
  }
  sherpa_ncnn::ModelConfig config;

  config.tokens = argv[1];
  config.encoder_param = argv[2];
  config.encoder_bin = argv[3];
  config.decoder_param = argv[4];
  config.decoder_bin = argv[5];
  config.joiner_param = argv[6];
  config.joiner_bin = argv[7];

  std::string wav_filename = argv[8];

  int32_t num_threads = 4;
  if (argc == 10) {
    num_threads = atoi(argv[9]);
  }
  config.encoder_opt.num_threads = num_threads;
  config.decoder_opt.num_threads = num_threads;
  config.joiner_opt.num_threads = num_threads;

  float expected_sampling_rate = 16000;

  sherpa_ncnn::SymbolTable sym(config.tokens);

  std::cout << config.ToString() << "\n";

  auto model = sherpa_ncnn::Model::Create(config);
  if (!model) {
    std::cout << "Failed to create a model\n";
    exit(EXIT_FAILURE);
  }

  bool is_ok = false;
  std::vector<float> samples =
      sherpa_ncnn::ReadWave(wav_filename, expected_sampling_rate, &is_ok);
  if (!is_ok) {
    fprintf(stderr, "Failed to read %s\n", wav_filename.c_str());
    exit(-1);
  }

  float duration = samples.size() / expected_sampling_rate;

  std::cout << "wav filename: " << wav_filename << "\n";
  std::cout << "wav duration (s): " << duration << "\n";

  knf::FbankOptions fbank_opts;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.snip_edges = false;
  fbank_opts.frame_opts.samp_freq = expected_sampling_rate;
  fbank_opts.mel_opts.num_bins = 80;

  sherpa_ncnn::FeatureExtractor feature_extractor(fbank_opts);
  feature_extractor.AcceptWaveform(expected_sampling_rate, samples.data(),
                                   samples.size());

  std::vector<float> tail_paddings(
      static_cast<int>(0.3 * expected_sampling_rate));
  feature_extractor.AcceptWaveform(expected_sampling_rate, tail_paddings.data(),
                                   tail_paddings.size());

  feature_extractor.InputFinished();

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

  std::vector<ncnn::Mat> states;
  ncnn::Mat encoder_out;

  int32_t num_processed = 0;
  while (feature_extractor.NumFramesReady() - num_processed >= segment) {
    ncnn::Mat features = feature_extractor.GetFrames(num_processed, segment);
    num_processed += offset;

    std::tie(encoder_out, states) = model->RunEncoder(features, states);

    GreedySearch(model.get(), encoder_out, &decoder_out, &hyp);
  }

  std::string text;
  for (int32_t i = context_size; i != hyp.size(); ++i) {
    text += sym[hyp[i]];
  }

  std::cout << "Recognition result for " << wav_filename << "\n"
            << text << "\n";

  return 0;
}
