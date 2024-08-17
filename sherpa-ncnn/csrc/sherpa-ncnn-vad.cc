/**
 * Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include <iostream>

#include "sherpa-ncnn/csrc/file-utils.h"
#include "sherpa-ncnn/csrc/voice-activity-detector.h"
#include "sherpa-ncnn/csrc/wave-reader.h"
#include "sherpa-ncnn/csrc/wave-writer.h"

int main() {
  std::string usage = R"usage(
This file shows how to use silero vad to remove silences from a file.

===========Usage============:

0. Build sherpa-ncnn
--------------------

mkdir -p $HOME/open-source
cd $HOME/open-source
git clone https://github.com/k2-fsa/sherpa-ncnn
cd sherpa-ncnn
mkdir build
cd build
cmake ..
make -j3

1. Download the vad model
-------------------------

cd $HOME/open-source/sherpa-ncnn/build
wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/sherpa-ncnn-silero-vad.tar.bz2
tar xvf sherpa-ncnn-silero-vad.tar.bz2

2. Download the test data
-------------------------

cd $HOME/open-source/sherpa-ncnn/build
wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/lei-jun-test.wav
wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/Obama.wav

3. Run it!
----------

cd $HOME/open-source/sherpa-ncnn/build
./bin/sherpa-ncnn-vad

**Note**: We only support 16000Hz wav files.
  )usage";

  sherpa_ncnn::SileroVadModelConfig config;
  config.sample_rate = 16000;
  config.param = "./sherpa-ncnn-silero-vad/silero.ncnn.param";
  config.bin = "./sherpa-ncnn-silero-vad/silero.ncnn.bin";
  config.window_size = 512;

  if (!config.Validate()) {
    fprintf(stderr, "%s %d: %s", __FILE__, static_cast<int32_t>(__LINE__),
            usage.c_str());
    return -1;
  }

  std::string input_wave = "./lei-jun-test.wav";
  // std::string input_wave = "./Obama.wav";
  if (!sherpa_ncnn::FileExists(input_wave)) {
    fprintf(stderr, "%s %d: %s", __FILE__, static_cast<int32_t>(__LINE__),
            usage.c_str());
    return -1;
  }

  bool is_ok = false;
  std::vector<float> samples =
      sherpa_ncnn::ReadWave(input_wave, config.sample_rate, &is_ok);
  if (!is_ok) {
    fprintf(stderr, "%s %d: We support only %d wave files", __FILE__,
            static_cast<int32_t>(__LINE__), config.sample_rate);
    return -1;
  }

  sherpa_ncnn::VoiceActivityDetector vad(config);
  int32_t num_samples = static_cast<int32_t>(samples.size());

  std::vector<sherpa_ncnn::SpeechSegment> segments;

  for (int32_t i = 0; i < samples.size(); i += config.window_size) {
    vad.AcceptWaveform(samples.data() + i, config.window_size);
    while (!vad.Empty()) {
      const auto &front = vad.Front();
      segments.push_back(front);

      vad.Pop();
    }
  }

  vad.Flush();
  while (!vad.Empty()) {
    const auto &front = vad.Front();
    segments.push_back(front);

    vad.Pop();
  }

  std::vector<float> all_samples;
  for (const auto &s : segments) {
    float start = s.start / static_cast<float>(config.sample_rate);
    float duration = s.samples.size() / static_cast<float>(config.sample_rate);
    float stop = start + duration;  // in seconds
                                    //
    fprintf(stderr, "%.3f -- %.3f s\n", start, start + duration);
    all_samples.insert(all_samples.end(), s.samples.begin(), s.samples.end());
  }

  std::string out_wave = "./out-without-silence.wav";
  is_ok = sherpa_ncnn::WriteWave(out_wave, config.sample_rate,
                                 all_samples.data(), all_samples.size());
  if (is_ok) {
    fprintf(stderr, "Saved to %s\n", out_wave.c_str());
  } else {
    fprintf(stderr, "Failed to saved to %s\n", out_wave.c_str());
  }

  return 0;
}
