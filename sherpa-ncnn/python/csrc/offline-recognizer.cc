// sherpa-ncnn/python/csrc/offline-recognizer.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/python/csrc/offline-recognizer.h"

#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/offline-recognizer.h"
#include "sherpa-ncnn/python/csrc/offline-model-config.h"

namespace sherpa_ncnn {

static void PybindOfflineRecognizerConfig(py::module *m) {
  PybindOfflineModelConfig(m);

  using PyClass = OfflineRecognizerConfig;
  py::class_<PyClass>(*m, "OfflineRecognizerConfig")
      .def(py::init<const FeatureExtractorConfig &, const OfflineModelConfig &,
                    const std::string &, float>(),
           py::arg("feat_config") = FeatureExtractorConfig(),
           py::arg("model_config") = OfflineModelConfig(),
           py::arg("decoding_method") = "greedy_search",
           py::arg("blank_penalty") = 0.0)
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("decoding_method", &PyClass::decoding_method)
      .def_readwrite("blank_penalty", &PyClass::blank_penalty)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineRecognizer(py::module *m) {
  PybindOfflineRecognizerConfig(m);

  using PyClass = OfflineRecognizer;
  py::class_<PyClass>(*m, "OfflineRecognizer")
      .def(py::init<const OfflineRecognizerConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("config", &PyClass::GetConfig)
      .def(
          "create_stream",
          [](const PyClass &self) { return self.CreateStream(); },
          py::call_guard<py::gil_scoped_release>())
      .def("decode_stream", &PyClass::DecodeStream, py::arg("s"),
           py::call_guard<py::gil_scoped_release>())
      .def("set_config", &PyClass::SetConfig, py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "decode_streams",
          [](const PyClass &self, std::vector<OfflineStream *> ss) {
            self.DecodeStreams(ss.data(), ss.size());
          },
          py::arg("ss"), py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_ncnn
