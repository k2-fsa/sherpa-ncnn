// sherpa-ncnn/python/csrc/offline-sense-voice-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-sense-voice-model-config.h"

#include <string>
#include <vector>

#include "sherpa-ncnn/python/csrc/offline-sense-voice-model-config.h"

namespace sherpa_ncnn {

void PybindOfflineSenseVoiceModelConfig(py::module *m) {
  using PyClass = OfflineSenseVoiceModelConfig;
  py::class_<PyClass>(*m, "OfflineSenseVoiceModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &, const std::string &, bool>(),
           py::arg("model_dir"), py::arg("language") = "auto",
           py::arg("use_itn") = true)
      .def_readwrite("model_dir", &PyClass::model_dir)
      .def_readwrite("language", &PyClass::language)
      .def_readwrite("use_itn", &PyClass::use_itn)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_ncnn
