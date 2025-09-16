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
#include <vector>

#include "sherpa-ncnn/csrc/file-utils.h"
#include "sherpa-ncnn/csrc/macros.h"

namespace sherpa_ncnn {

void SileroVadModelConfig::Register(ParseOptions *po) {
  po->Register("silero-vad-model-dir", &model_dir,
               "Path to SileroVad model directory. It should contain "
               "silero.ncnn.param and silero.ncnn.bin.");

  po->Register("silero-vad-threshold", &threshold, "VAD Threshold");
  po->Register("silero-vad-num-threads", &num_threads,
               "Number of threads to run the model");
}

bool SileroVadModelConfig::Validate() const {
  if (model_dir.empty()) {
    SHERPA_NCNN_LOGE("Please provide --silero-vad-model-dir");
    return false;
  }

  std::vector<std::string> files_to_check = {
      "silero.ncnn.param",
      "silero.ncnn.bin",
  };

  bool ok = true;
  for (const auto &f : files_to_check) {
    auto name = model_dir + "/" + f;
    if (!FileExists(name)) {
      SHERPA_NCNN_LOGE("'%s' does not exist inside the directory '%s'",
                       name.c_str(), model_dir.c_str());
      ok = false;
    }
  }

  if (!ok) {
    return false;
  }

  if (threshold < 0.01) {
    SHERPA_NCNN_LOGE("Please use a larger value for threshold. Given: %f",
                     threshold);
    return false;
  }

  if (threshold >= 1) {
    SHERPA_NCNN_LOGE("Please use a smaller value for threshold. Given: %f",
                     threshold);
    return false;
  }

  if (num_threads < 1) {
    SHERPA_NCNN_LOGE("Please use a larger num_threads. Current: %d",
                     num_threads);
    return false;
  }

  return true;
}

std::string SileroVadModelConfig::ToString() const {
  std::ostringstream os;

  os << "SileroVadModelConfig(";
  os << "model_dir=\"" << model_dir << "\", ";
  os << "threshold=" << threshold << ", ";
  os << "min_silence_duration=" << min_silence_duration << ", ";
  os << "min_speech_duration=" << min_speech_duration << ", ";
  os << "window_size=" << window_size << ", ";
  os << "sample_rate=" << sample_rate << ", ";
  os << "use_vulkan_compute=" << (use_vulkan_compute ? "True" : "False")
     << ", ";
  os << "num_threads=" << num_threads << ")";

  return os.str();
}

}  // namespace sherpa_ncnn
