/**
 * Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "sherpa-ncnn/csrc/file-utils.h"

#include <fstream>
#include <string>

#include "sherpa-ncnn/csrc/macros.h"

namespace sherpa_ncnn {

bool FileExists(const std::string &filename) {
  return std::ifstream(filename).good();
}

void AssertFileExists(const std::string &filename) {
  if (!FileExists(filename)) {
    NCNN_LOGE("filename '%s' does not exist", filename.c_str());
    SHERPA_NCNN_EXIT(-1);
  }
}

}  // namespace sherpa_ncnn
