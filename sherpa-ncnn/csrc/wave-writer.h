/**
 * Copyright (c)  2022-2024  Xiaomi Corporation (authors: Fangjun Kuang)
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
#ifndef SHERPA_NCNN_CSRC_WAVE_WRITER_H_
#define SHERPA_NCNN_CSRC_WAVE_WRITER_H_

#include <cstdint>
#include <string>

namespace sherpa_ncnn {

// Write a single channel wave file.
// Note that the input samples are in the range [-1, 1]. It will be multiplied
// by 32767 and saved in int16_t format in the wave file.
//
// @param filename Path to save the samples.
// @param sampling_rate Sample rate of the samples.
// @param samples Pointer to the samples
// @param n Number of samples
// @return Return true if the write succeeds; return false otherwise.
bool WriteWave(const std::string &filename, int32_t sampling_rate,
               const float *samples, int32_t n);

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_WAVE_WRITER_H_
