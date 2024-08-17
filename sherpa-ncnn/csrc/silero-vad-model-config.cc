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
#include "sherpa-ncnn/csrc/silero-vad-model-config.h"

#include <sstream>
#include <string>

#include "platform.h"  // for NCNN_LOGE, NOLINT
#include "sherpa-ncnn/csrc/file-utils.h"

namespace sherpa_ncnn {

bool SileroVadModelConfig::Validate() const {
  if (param.empty()) {
    NCNN_LOGE("Please provide filename to silero.ncnn.param");
    return false;
  }

  if (!FileExists(param)) {
    NCNN_LOGE("'%s' does not exist", param.c_str());
    return false;
  }

  if (bin.empty()) {
    NCNN_LOGE("Please provide filename to silero.ncnn.bin");
    return false;
  }

  if (!FileExists(bin)) {
    NCNN_LOGE("'%s' does not exist", bin.c_str());
    return false;
  }

  if (threshold < 0.01) {
    NCNN_LOGE("Please use a larger value for threshold. Given: %f", threshold);
    return false;
  }

  if (threshold >= 1) {
    NCNN_LOGE("Please use a smaller value for threshold. Given: %f", threshold);
    return false;
  }

  return true;
}

std::string SileroVadModelConfig::ToString() const {
  std::ostringstream os;

  os << "SilerVadModelConfig(";
  os << "param=\"" << param << "\", ";
  os << "bin=\"" << bin << "\", ";
  os << "threshold=" << threshold << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "min_speech_duration=" << min_speech_duration << ", ";
  os << "window_size=" << window_size << ", ";
  os << "sample_rate=" << sample_rate << ", ";
  os << "use_vulkan_compute=" << (use_vulkan_compute ? "True" : "False")
     << ", ";
  os << "num_threads=" << opt.num_threads << ")";

  return os.str();
}

}  // namespace sherpa_ncnn
