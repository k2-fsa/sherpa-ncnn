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

#include <stdio.h>

#include <fstream>
#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/resample.h"

int32_t main(int32_t argc, char *argv[]) {
  const char *kUsage = R"(
Usage:

  ./bin/test-resample in.raw in_sample_rate out.raw out_sample_rate

where

 - in.raw, containing input raw PCM samples, mono, 16-bit
 - in_sample_rate, sample rate for in.raw
 - out.raw, containing output raw PCM samples, mono, 16-bit
 - out_sample_rate, sample rate for out.raw

For instance, if in_sample_rate is 48000 and out_sample_rate is 16000,
you can use

  sox -t raw -r 48000 -e signed -b 16 -c 1 in.raw in.wav
  sox -t raw -r 16000 -e signed -b 16 -c 1 out.raw out.wav

  soxi in.wav
  soxi out.wav

You can compare the number of samples in in.wav and out.wav.
Also, you can play a.wav and b.wav.

  )";
  if (argc != 5) {
    fprintf(stderr, "%s", kUsage);
    exit(-1);
  }

  std::string in_raw = argv[1];
  int32_t in_sample_rate = atoi(argv[2]);

  std::string out_raw = argv[3];
  int32_t out_sample_rate = atoi(argv[4]);
  fprintf(stderr, "in sample rate: %d, out_sample_rate: %d\n", in_sample_rate,
          out_sample_rate);
  fprintf(stderr, "in_raw : %s, out_raw: %s\n", in_raw.c_str(),
          out_raw.c_str());

  std::ifstream is(in_raw, std::ios::binary);
  std::vector<int8_t> buffer(std::istreambuf_iterator<char>(is), {});
  if (buffer.size() % 2 != 0) {
    fprintf(stderr, "expect int16 samples\n");
    exit(-1);
  }

  int32_t num_samples = buffer.size() / 2;
  fprintf(stderr, "num_samples: %d\n", num_samples);
  const int16_t *p = reinterpret_cast<int16_t *>(buffer.data());

  std::vector<float> in_float(buffer.size() / 2);
  for (int32_t i = 0; i != num_samples; ++i) {
    in_float[i] = p[i] / 32768.0f;
  }

  float min_freq = std::min(in_sample_rate, out_sample_rate);
  float lowpass_cutoff = 0.99 * 0.5 * min_freq;

  int32_t lowpass_filter_width = 6;
  sherpa_ncnn::LinearResample resampler(in_sample_rate, out_sample_rate,
                                        lowpass_cutoff, lowpass_filter_width);

  // simulate streaming
  int32_t chunk = 100;
  const float *q = in_float.data();

  std::vector<float> out_float;

  int32_t start = 0;
  for (start = 0; start + chunk < num_samples; start += chunk) {
    std::vector<float> tmp;
    resampler.Resample(q, chunk, false, &tmp);
    out_float.insert(out_float.end(), tmp.begin(), tmp.end());
    q += chunk;
  }

  std::vector<float> tmp;
  resampler.Resample(q, num_samples - start, true, &tmp);
  out_float.insert(out_float.end(), tmp.begin(), tmp.end());

  std::vector<int16_t> out_short(out_float.size());
  int32_t num_out_samples = out_float.size();
  for (int32_t i = 0; i != num_out_samples; ++i) {
    out_short[i] = std::min(32767, static_cast<int32_t>(out_float[i] * 32767));
  }

  std::ofstream os(out_raw, std::ios::binary);
  os.write(reinterpret_cast<char *>(out_short.data()),
           out_short.size() * sizeof(int16_t));

  return 0;
}
