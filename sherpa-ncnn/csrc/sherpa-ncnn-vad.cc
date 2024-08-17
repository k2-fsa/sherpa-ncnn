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

#include "sherpa-ncnn/csrc/silero-vad-model.h"

int main() {
  sherpa_ncnn::SileroVadModelConfig config;
  config.param = "./silero.ncnn.param";
  config.bin = "./silero.ncnn.bin";
  if (!config.Validate()) {
    return -1;
  }

  sherpa_ncnn::SileroVadModel model(config);

  std::cout << config.ToString() << "\n";
  return 0;
}
