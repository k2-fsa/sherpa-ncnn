// sherpa-ncnn/python/csrc/offline-tts-vits-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/python/csrc/offline-tts-vits-model-config.h"

#include <string>

#include "sherpa-ncnn/csrc/offline-tts-vits-model-config.h"

namespace sherpa_ncnn {

void PybindOfflineTtsVitsModelConfig(py::module *m) {
  using PyClass = OfflineTtsVitsModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsVitsModelConfig")
      .def(py::init<>())
      .def(py::init<const std::string &>(), py::arg("model_dir") = "")
      .def_readwrite("model_dir", &PyClass::model_dir)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_ncnn
