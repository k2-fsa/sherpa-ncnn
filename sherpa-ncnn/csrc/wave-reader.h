/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

#ifndef SHERPA_NCNN_CSRC_WAVE_READER_H_
#define SHERPA_NCNN_CSRC_WAVE_READER_H_

#include <istream>
#include <string>
#include <vector>

namespace sherpa_ncnn {

/** Read a wave file with expected sample rate.

    @param filename Path to a wave file. It MUST be single channel, PCM encoded.
    @param expected_sample_rate  Expected sample rate of the wave file. If the
                               sample rate don't match, it throws an exception.
    @param is_ok On return it is true if the reading succeeded; false otherwise.

    @return Return wave samples normalized to the range [-1, 1).
 */
std::vector<float> ReadWave(const std::string &filename,
                            float expected_sample_rate, bool *is_ok);

std::vector<float> ReadWave(std::istream &is, float expected_sample_rate,
                            bool *is_ok);

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_WAVE_READER_H_
