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

#include "sherpa-ncnn/csrc/recognizer.h"

namespace sherpa_ncnn {

static constexpr const char *kDecoderConfigInitDoc = R"doc(
Constructor for DecoderConfig.

Args:
  method:
    Decoding method. Supported values are: greedy_search, modified_beam_search.
  num_active_paths:
    Used only when method is modified_beam_search. It specifies the number of
    actives paths during beam search.
  enable_endpoint:
    True to enable endpoint detection. False to disable endpoint detection.
  endpoint_config:
    Used only when ``enable_endpoint`` is True.
)doc";

static void PybindRecognitionResult(py::module *m) {
  using PyClass = RecognitionResult;
  py::class_<PyClass>(*m, "RecognitionResult")
      .def_property_readonly(
          "text", [](PyClass &self) -> std::string { return self.text; });
}

static void PybindDecoderConfig(py::module *m) {
  using PyClass = DecoderConfig;
  py::class_<PyClass>(*m, "DecoderConfig")
      .def(py::init<const std::string &, int32_t, bool,
                    const EndpointConfig &>(),
           py::arg("method"), py::arg("num_active_paths"),
           py::arg("enable_endpoint"), py::arg("endpoint_config"),
           kDecoderConfigInitDoc)
      .def("__str__", &PyClass::ToString)
      .def_property_readonly("method",
                             [](const PyClass &self) { return self.method; })
      .def_property_readonly(
          "num_active_paths",
          [](const PyClass &self) { return self.num_active_paths; })
      .def_property_readonly(
          "enable_endpoint",
          [](const PyClass &self) { return self.enable_endpoint; })
      .def_property_readonly("endpoint_config", [](const PyClass &self) {
        return self.endpoint_config;
      });
}

void PybindRecognizer(py::module *m) {
  PybindRecognitionResult(m);
  PybindDecoderConfig(m);

  using PyClass = Recognizer;
  py::class_<PyClass>(*m, "Recognizer")
      .def(py::init([](const DecoderConfig &decoder_config,
                       const ModelConfig &model_config,
                       float sample_rate = 16000) -> std::unique_ptr<PyClass> {
             knf::FbankOptions fbank_opts;
             fbank_opts.frame_opts.dither = 0;
             fbank_opts.frame_opts.snip_edges = false;
             fbank_opts.frame_opts.samp_freq = sample_rate;
             fbank_opts.mel_opts.num_bins = 80;

             return std::make_unique<PyClass>(decoder_config, model_config,
                                              fbank_opts);
           }),
           py::arg("decoder_config"), py::arg("model_config"),
           py::arg("sample_rate") = 16000)
      .def("accept_waveform",
           [](PyClass &self, float sample_rate, py::array_t<float> waveform) {
             self.AcceptWaveform(sample_rate, waveform.data(), waveform.size());
           })
      .def("input_finished", &PyClass::InputFinished)
      .def("decode", &PyClass::Decode)
      .def_property_readonly("result",
                             [](PyClass &self) { return self.GetResult(); })
      .def("is_endpoint", &PyClass::IsEndpoint)
      .def("reset", &PyClass::Reset);
}

}  // namespace sherpa_ncnn
