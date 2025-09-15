// sherpa-ncnn/python/csrc/offline-model-config.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-ncnn/python/csrc/offline-model-config.h"

#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/offline-model-config.h"
#include "sherpa-ncnn/python/csrc/offline-sense-voice-model-config.h"

namespace sherpa_ncnn {

void PybindOfflineModelConfig(py::module *m) {
  PybindOfflineSenseVoiceModelConfig(m);

  using PyClass = OfflineModelConfig;
  py::class_<PyClass>(*m, "OfflineModelConfig")
      .def(py::init<const OfflineSenseVoiceModelConfig &, const std::string &,
                    int32_t, bool>(),
           py::arg("sense_voice") = OfflineSenseVoiceModelConfig(),
           py::arg("tokens") = "", py::arg("num_threads") = 1,
           py::arg("debug") = false)
      .def_readwrite("sense_voice", &PyClass::sense_voice)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_ncnn
