/**
 * Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "sherpa-ncnn/python/csrc/recognizer.h"

#include <memory>
#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/recognizer.h"

namespace sherpa_ncnn {

static constexpr const char *kRecognizerConfigInitDoc = R"doc(
Constructor for RecognizerConfig.

Args:
  feat_config:
    Feature extraction options.
  model_config:
    Config for NN models.
  decoder_config:
    Config for decoding.
  endpoint_config:
    Config for endpointing
  enable_endpoint:
    True to enable endpoint detection. False to disable endpoint detection.
)doc";

static void PybindRecognitionResult(py::module *m) {
  using PyClass = RecognitionResult;
  py::class_<PyClass>(*m, "RecognitionResult")
      .def_property_readonly(
          "text", [](PyClass &self) -> std::string { return self.text; })
      .def_property_readonly(
          "tokens",
          [](PyClass &self) -> std::vector<int> { return self.tokens; })
      .def_property_readonly("stokens",
                             [](PyClass &self) -> std::vector<std::string> {
                               return self.stokens;
                             })
      .def_property_readonly(
          "timestamps",
          [](PyClass &self) -> std::vector<float> { return self.timestamps; });
}

static void PybindRecognizerConfig(py::module *m) {
  using PyClass = RecognizerConfig;
  py::class_<PyClass>(*m, "RecognizerConfig")
      .def(py::init<const FeatureExtractorConfig &, const ModelConfig &,
                    const DecoderConfig &, const EndpointConfig &, bool,
                    const std::string &, float>(),
           py::arg("feat_config"), py::arg("model_config"),
           py::arg("decoder_config"), py::arg("endpoint_config"),
           py::arg("enable_endpoint"), py::arg("hotwords_file") = "",
           py::arg("hotwords_score") = 1.5, kRecognizerConfigInitDoc)
      .def("__str__", &PyClass::ToString)
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("decoder_config", &PyClass::decoder_config)
      .def_readwrite("endpoint_config", &PyClass::endpoint_config)
      .def_readwrite("enable_endpoint", &PyClass::enable_endpoint)
      .def_readwrite("hotwords_file", &PyClass::hotwords_file)
      .def_readwrite("hotwords_score", &PyClass::hotwords_score);
}

void PybindRecognizer(py::module *m) {
  PybindRecognitionResult(m);
  PybindRecognizerConfig(m);

  using PyClass = Recognizer;
  py::class_<PyClass>(*m, "Recognizer")
      .def(py::init<const RecognizerConfig &>(), py::arg("config"))
      .def("create_stream", &PyClass::CreateStream)
      .def("decode_stream", &PyClass::DecodeStream, py::arg("s"))
      .def("is_ready", &PyClass::IsReady, py::arg("s"))
      .def("reset", &PyClass::Reset, py::arg("s"))
      .def("is_endpoint", &PyClass::IsEndpoint, py::arg("s"))
      .def("get_result", &PyClass::GetResult, py::arg("s"));
}

}  // namespace sherpa_ncnn
