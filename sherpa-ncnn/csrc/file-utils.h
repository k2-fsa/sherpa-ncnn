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
#ifndef SHERPA_NCNN_CSRC_FILE_UTILS_H_
#define SHERPA_NCNN_CSRC_FILE_UTILS_H_

#include <fstream>
#include <string>

#include "platform.h"  // for NCNN_LOGE, NOLINT

namespace sherpa_ncnn {

/** Check whether a given path is a file or not
 *
 * @param filename Path to check.
 * @return Return true if the given path is a file; return false otherwise.
 */
bool FileExists(const std::string &filename);

/** Abort if the file does not exist.
 *
 * @param filename The file to check.
 */
void AssertFileExists(const std::string &filename);

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_FILE_UTILS_H_
