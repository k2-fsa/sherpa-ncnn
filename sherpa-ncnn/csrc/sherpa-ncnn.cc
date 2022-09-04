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

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "net.h"
#include "sherpa-ncnn/csrc/symbol-table.h"
#include "sherpa-ncnn/csrc/wave-reader.h"
#include <algorithm>
#include <iostream>

static void InitNet(ncnn::Net &net, const std::string &param,
                    const std::string &model) {
  if (net.load_param(param.c_str())) {
    std::cerr << "failed to load " << param << "\n";
    exit(-1);
  }

  if (net.load_model(model.c_str())) {
    std::cerr << "failed to load " << model << "\n";
    exit(-1);
  }
}

int main() {

  std::string encoder_param =
      "bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param";

  std::string encoder_model =
      "bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin";

  std::string decoder_param =
      "bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param";

  std::string decoder_model =
      "bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin";

  std::string joiner_param =
      "bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param";

  std::string joiner_model =
      "bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin";

  std::string wav1 = "./test_wavs/1089-134686-0001.wav";
  // wav1 = "./test_wavs/1221-135766-0001.wav";
  wav1 = "./test_wavs/1221-135766-0002.wav";

  ncnn::Net encoder_net;
  encoder_net.opt.use_packing_layout = false;
  encoder_net.opt.use_fp16_storage = false;

  ncnn::Net decoder_net;
  decoder_net.opt.use_packing_layout = false;

  ncnn::Net joiner_net;
  joiner_net.opt.use_packing_layout = false;

  InitNet(encoder_net, encoder_param, encoder_model);
  InitNet(decoder_net, decoder_param, decoder_model);
  InitNet(joiner_net, joiner_param, joiner_model);

  std::vector<float> samples = sherpa_ncnn::ReadWave(wav1, 16000);

  knf::FbankOptions opts;
  opts.frame_opts.dither = 0;
  opts.frame_opts.snip_edges = false;
  opts.frame_opts.samp_freq = 16000;

  opts.mel_opts.num_bins = 80;

  knf::OnlineFbank fbank(opts);
  fbank.AcceptWaveform(16000, samples.data(), samples.size());
  fbank.InputFinished();

  int32_t num_encoder_layers = 12;
  int32_t batch_size = 1;
  int32_t d_model = 512;
  int32_t rnn_hidden_size = 1024;

  ncnn::Mat h0;
  h0.create(d_model, num_encoder_layers);
  ncnn::Mat c0;
  c0.create(rnn_hidden_size, num_encoder_layers);
  h0.fill(0);
  c0.fill(0);

  int32_t feature_dim = 80;
  ncnn::Mat features;
  features.create(feature_dim, fbank.NumFramesReady());

  for (int32_t i = 0; i != fbank.NumFramesReady(); ++i) {
    const float *f = fbank.GetFrame(i);
    std::copy(f, f + feature_dim, features.row(i));
  }

  ncnn::Mat feature_lengths(1);
  feature_lengths[0] = features.h;

  ncnn::Extractor encoder_ex = encoder_net.create_extractor();

  encoder_ex.input("in0", features);
  encoder_ex.input("in1", feature_lengths);
  encoder_ex.input("in2", h0);
  encoder_ex.input("in3", c0);

  ncnn::Mat encoder_out;
  encoder_ex.extract("out0", encoder_out);

  int32_t context_size = 2;
  int32_t blank_id = 0;

  std::vector<int32_t> hyp(context_size, blank_id);
  ncnn::Mat decoder_input(context_size);
  static_cast<int32_t *>(decoder_input)[0] = blank_id + 1;
  static_cast<int32_t *>(decoder_input)[1] = blank_id + 2;
  decoder_input.fill(blank_id);

  ncnn::Extractor decoder_ex = decoder_net.create_extractor();
  ncnn::Mat decoder_out;
  decoder_ex.input("in0", decoder_input);
  decoder_ex.extract("out0", decoder_out);
  decoder_out = decoder_out.reshape(decoder_out.w);

  ncnn::Mat joiner_out;
  for (int32_t t = 0; t != encoder_out.h; ++t) {
    ncnn::Mat encoder_out_t(512, encoder_out.row(t));

    auto joiner_ex = joiner_net.create_extractor();
    joiner_ex.input("in0", encoder_out_t);
    joiner_ex.input("in1", decoder_out);

    joiner_ex.extract("out0", joiner_out);

    auto y = static_cast<int32_t>(
        std::distance(static_cast<const float *>(joiner_out),
                      std::max_element(static_cast<const float *>(joiner_out),
                                       static_cast<const float *>(joiner_out) +
                                           joiner_out.w)));

    if (y != blank_id) {
      static_cast<int32_t *>(decoder_input)[0] = hyp.back();
      static_cast<int32_t *>(decoder_input)[1] = y;
      hyp.push_back(y);

      decoder_ex = decoder_net.create_extractor();
      decoder_ex.input("in0", decoder_input);
      decoder_ex.extract("out0", decoder_out);
      decoder_out = decoder_out.reshape(decoder_out.w);
    }
  }
  std::string text;
  sherpa_ncnn::SymbolTable sym("./tokens.txt");
  for (int32_t i = context_size; i != hyp.size(); ++i) {
    text += sym[hyp[i]];
  }

  fprintf(stderr, "%s\n", text.c_str());
  return 0;
}
