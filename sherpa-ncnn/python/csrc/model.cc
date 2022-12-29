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
#include "sherpa-ncnn/python/csrc/mat-util.h"

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
)doc";

static void PybindModelConfig(py::module *m) {
  using PyClass = ModelConfig;
  py::class_<PyClass>(*m, "ModelConfig")
      .def(py::init([](const std::string &encoder_param,
                       const std::string &encoder_bin,
                       const std::string &decoder_param,
                       const std::string &decoder_bin,
                       const std::string &joiner_param,
                       const std::string &joiner_bin,
                       int32_t num_threads) -> std::unique_ptr<PyClass> {
             auto ans = std::make_unique<PyClass>();
             ans->encoder_param = encoder_param;
             ans->encoder_bin = encoder_bin;
             ans->decoder_param = decoder_param;
             ans->decoder_bin = decoder_bin;
             ans->joiner_param = joiner_param;
             ans->joiner_bin = joiner_bin;

             ans->use_vulkan_compute = false;

             ans->encoder_opt.num_threads = num_threads;
             ans->decoder_opt.num_threads = num_threads;
             ans->joiner_opt.num_threads = num_threads;

             return ans;
           }),
           py::arg("encoder_param"), py::arg("encoder_bin"),
           py::arg("decoder_param"), py::arg("decoder_bin"),
           py::arg("joiner_param"), py::arg("joiner_bin"),
           py::arg("num_threads"), kModelConfigInitDoc);
}

void PybindModel(py::module *m) {
  PybindModelConfig(m);

  using PyClass = Model;
  py::class_<PyClass>(*m, "Model")
      .def_static("create", &PyClass::Create, py::arg("config"))
      .def(
          "run_encoder",
          [](PyClass &self, py::array _features,
             const std::vector<py::array> &_states)
              -> std::pair<py::array, std::vector<py::array>> {
            ncnn::Mat features = ArrayToMat(_features);

            std::vector<ncnn::Mat> states;
            states.reserve(_states.size());
            for (const auto &s : _states) {
              states.push_back(ArrayToMat(s));
            }

            ncnn::Mat encoder_out;
            std::vector<ncnn::Mat> _next_states;

            std::tie(encoder_out, _next_states) =
                self.RunEncoder(features, states);

            std::vector<py::array> next_states;
            next_states.reserve(_next_states.size());
            for (const auto &m : _next_states) {
              next_states.push_back(MatToArray(m));
            }

            return std::make_pair(MatToArray(encoder_out), next_states);
          },
          py::arg("features"), py::arg("states"))
      .def(
          "run_decoder",
          [](PyClass &self, py::array _decoder_input) -> py::array {
            ncnn::Mat decoder_input = ArrayToMat(_decoder_input);
            ncnn::Mat decoder_out = self.RunDecoder(decoder_input);
            return MatToArray(decoder_out);
          },
          py::arg("decoder_input"))
      .def(
          "run_joiner",
          [](PyClass &self, py::array _encoder_out,
             py::array _decoder_out) -> py::array {
            ncnn::Mat encoder_out = ArrayToMat(_encoder_out);
            ncnn::Mat decoder_out = ArrayToMat(_decoder_out);
            ncnn::Mat joiner_out = self.RunJoiner(encoder_out, decoder_out);

            return MatToArray(joiner_out);
          },
          py::arg("encoder_out"), py::arg("decoder_out"))
      .def_property_readonly("context_size", &PyClass::ContextSize)
      .def_property_readonly("blank_id", &PyClass::BlankId)
      .def_property_readonly("segment", &PyClass::Segment)
      .def_property_readonly("offset", &PyClass::Offset);
}

}  // namespace sherpa_ncnn
