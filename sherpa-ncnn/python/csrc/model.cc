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

#include "sherpa-ncnn/python/csrc/model.h"

#include <memory>
#include <string>

#include "sherpa-ncnn/csrc/model.h"

namespace sherpa_ncnn {

const char *kModelConfigInitDoc = R"doc(
Constructor for ModelConfig.

Please refer to
`<https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html>`_
for download links about pre-trained models.

Args:
  encoder_param:
    Path to encoder.ncnn.param.
  encoder_bin:
    Path to encoder.ncnn.bin.
  decoder_param:
    Path to decoder.ncnn.param.
  decoder_bin:
    Path to decoder.ncnn.bin.
  joiner_param:
    Path to joiner.ncnn.param.
  joiner_bin:
    Path to joiner.ncnn.bin.
  num_threads:
    Number of threads to use for neural network computation.
  tokens:
    Path to tokens.txt
)doc";

static void PybindModelConfig(py::module *m) {
  using PyClass = ModelConfig;
  py::class_<PyClass>(*m, "ModelConfig")
      .def(py::init([](const std::string &encoder_param,
                       const std::string &encoder_bin,
                       const std::string &decoder_param,
                       const std::string &decoder_bin,
                       const std::string &joiner_param,
                       const std::string &joiner_bin, int32_t num_threads,
                       const std::string &tokens) -> std::unique_ptr<PyClass> {
             auto ans = std::make_unique<PyClass>();
             ans->encoder_param = encoder_param;
             ans->encoder_bin = encoder_bin;
             ans->decoder_param = decoder_param;
             ans->decoder_bin = decoder_bin;
             ans->joiner_param = joiner_param;
             ans->joiner_bin = joiner_bin;
             ans->tokens = tokens;

             ans->use_vulkan_compute = false;

             ans->encoder_opt.num_threads = num_threads;
             ans->decoder_opt.num_threads = num_threads;
             ans->joiner_opt.num_threads = num_threads;

             return ans;
           }),
           py::arg("encoder_param"), py::arg("encoder_bin"),
           py::arg("decoder_param"), py::arg("decoder_bin"),
           py::arg("joiner_param"), py::arg("joiner_bin"),
           py::arg("num_threads"), py::arg("tokens"), kModelConfigInitDoc);
}

void PybindModel(py::module *m) { PybindModelConfig(m); }

}  // namespace sherpa_ncnn
