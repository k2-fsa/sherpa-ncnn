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
#include "sherpa-ncnn/csrc/symbol-table.h"
#include "sherpa-ncnn/csrc/wave-reader.h"

/** Compute fbank features of the input wave filename.
 *
 * @param wav_filename. Path to a mono wave file.
 * @param expected_sampling_rate  Expected sampling rate of the input wave file.
 * @return Return a mat of shape (num_frames, feature_dim).
 *         Note: ans.w == feature_dim; ans.h == num_frames
 *
 */
static ncnn::Mat ComputeFeatures(const std::string &wav_filename,
                                 float expected_sampling_rate) {
  std::vector<float> samples =
      sherpa_ncnn::ReadWave(wav_filename, expected_sampling_rate);

  float duration = samples.size() / expected_sampling_rate;

  std::cout << "wav filename: " << wav_filename << "\n";
  std::cout << "wav duration (s): " << duration << "\n";

  knf::FbankOptions opts;
  opts.frame_opts.dither = 0;
  opts.frame_opts.snip_edges = false;
  opts.frame_opts.samp_freq = expected_sampling_rate;

  opts.mel_opts.num_bins = 80;

  knf::OnlineFbank fbank(opts);
  fbank.AcceptWaveform(expected_sampling_rate, samples.data(), samples.size());
  fbank.InputFinished();

  int32_t feature_dim = 80;
  ncnn::Mat features;
  features.create(feature_dim, fbank.NumFramesReady());

  for (int32_t i = 0; i != fbank.NumFramesReady(); ++i) {
    const float *f = fbank.GetFrame(i);
    std::copy(f, f + feature_dim, features.row(i));
  }

  return features;
}

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

/** Run the encoder network.
 *
 * @param encoder_net The encoder model.
 * @param features  A 2-d mat of shape (num_frames, feature_dim).
 *                  Note: features.w = feature_dim.
 *                        features.h = num_frames.
 * @param num_threads  Number of threads to use for computation.
 *
 * @return Return the output of the encoder. Its shape is
 *  (num_frames, encoder_dim).
 *  Note: ans.w == encoder_dim; ans.h == num_frames
 */
static ncnn::Mat RunEncoder(ncnn::Net &encoder_net, ncnn::Mat &features,
                            int32_t num_threads) {
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

  ncnn::Mat feature_lengths(1);
  feature_lengths[0] = features.h;

  ncnn::Extractor encoder_ex = encoder_net.create_extractor();
  encoder_ex.set_num_threads(num_threads);

  encoder_ex.input("in0", features);
  encoder_ex.input("in1", feature_lengths);
  encoder_ex.input("in2", h0);
  encoder_ex.input("in3", c0);

  ncnn::Mat encoder_out;
  encoder_ex.extract("out0", encoder_out);

  return encoder_out;
}

/** Run the decoder network.
 *
 * @param decoder_net The decoder network.
 * @param  decoder_input A mat of shape (context_size,). Note: Its underlying
 *                       content consists of integers, though its type is float.
 * @param num_threads  Number of threads to use for computation.
 *
 * @return Return a mat of shape (decoder_dim,)
 */
static ncnn::Mat RunDecoder(ncnn::Net &decoder_net, ncnn::Mat &decoder_input,
                            int32_t num_threads) {
  ncnn::Extractor decoder_ex = decoder_net.create_extractor();
  decoder_ex.set_num_threads(num_threads);

  ncnn::Mat decoder_out;
  decoder_ex.input("in0", decoder_input);
  decoder_ex.extract("out0", decoder_out);
  decoder_out = decoder_out.reshape(decoder_out.w);

  return decoder_out;
}

/** Run the joiner network.
 *
 * @param joiner_net The joiner network.
 * @param encoder_out  A mat of shape (encoder_dim,)
 * @param decoder_out  A mat of shape (decoder_dim,)
 * @param num_threads  Number of threads to use for computation.
 *
 * @return Return the joiner output which is of shape (vocab_size,)
 */
static ncnn::Mat RunJoiner(ncnn::Net &joiner_net, ncnn::Mat &encoder_out,
                           ncnn::Mat &decoder_out, int32_t num_threads) {
  auto joiner_ex = joiner_net.create_extractor();
  joiner_ex.set_num_threads(num_threads);
  joiner_ex.input("in0", encoder_out);
  joiner_ex.input("in1", decoder_out);

  ncnn::Mat joiner_out;
  joiner_ex.extract("out0", joiner_out);
  return joiner_out;
}

int main(int argc, char *argv[]) {
  if (argc == 1 || argc > 3) {
    const char *usage = R"usage(
Usage:
  ./sherpa-ncnn /path/to/foo.wav [num_threads]

We assume that you have placed the models files in
the directory bar. That is, you should have the
following files:
  bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param
  bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin
  bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param
  bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin
  bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param
  bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin

We also assume that you have ./tokens.txt in the current directory.

You can find the above files in the following repository:

https://huggingface.co/csukuangfj/sherpa-ncnn-2022-09-05
)usage";
    std::cerr << usage << "\n";

    return 0;
  }

  sherpa_ncnn::SymbolTable sym("./tokens.txt");

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

  std::string wav = argv[1];
  int32_t num_threads = 5;
  if (argc == 3) {
    num_threads = atoi(argv[2]);
  }
  std::cout << "Number of threads: " << num_threads << "\n";

  ncnn::Mat features = ComputeFeatures(wav, 16000);

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

  ncnn::Mat encoder_out = RunEncoder(encoder_net, features, num_threads);

  int32_t context_size = 2;
  int32_t blank_id = 0;

  std::vector<int32_t> hyp(context_size, blank_id);
  ncnn::Mat decoder_input(context_size);
  static_cast<int32_t *>(decoder_input)[0] = blank_id + 1;
  static_cast<int32_t *>(decoder_input)[1] = blank_id + 2;
  decoder_input.fill(blank_id);

  ncnn::Mat decoder_out = RunDecoder(decoder_net, decoder_input, num_threads);

  for (int32_t t = 0; t != encoder_out.h; ++t) {
    ncnn::Mat encoder_out_t(512, encoder_out.row(t));
    ncnn::Mat joiner_out =
        RunJoiner(joiner_net, encoder_out_t, decoder_out, num_threads);

    auto y = static_cast<int32_t>(std::distance(
        static_cast<const float *>(joiner_out),
        std::max_element(
            static_cast<const float *>(joiner_out),
            static_cast<const float *>(joiner_out) + joiner_out.w)));

    if (y != blank_id) {
      static_cast<int32_t *>(decoder_input)[0] = hyp.back();
      static_cast<int32_t *>(decoder_input)[1] = y;
      hyp.push_back(y);

      decoder_out = RunDecoder(decoder_net, decoder_input, num_threads);
    }
  }
  std::string text;
  for (int32_t i = context_size; i != hyp.size(); ++i) {
    text += sym[hyp[i]];
  }

  std::cout << "Recognition result for " << wav << "\n" << text << "\n";
  return 0;
}
