// sherpa-ncnn/python/csrc/offline-tts-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/python/csrc/offline-tts-model-config.h"

#include <string>

#include "sherpa-ncnn/csrc/offline-tts-model-config.h"
#include "sherpa-ncnn/python/csrc/offline-tts-vits-model-config.h"

namespace sherpa_ncnn {

void PybindOfflineTtsModelConfig(py::module *m) {
  PybindOfflineTtsVitsModelConfig(m);

  using PyClass = OfflineTtsModelConfig;

  py::class_<PyClass>(*m, "OfflineTtsModelConfig")
      .def(py::init<>())
      .def(py::init<const OfflineTtsVitsModelConfig &, int32_t, bool>(),
           py::arg("vits") = OfflineTtsVitsModelConfig{},
           py::arg("num_threads") = 1, py::arg("debug") = false)
      .def_readwrite("vits", &PyClass::vits)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def("__str__", &PyClass::ToString)
      .def("validate", &PyClass::Validate);
}

}  // namespace sherpa_ncnn
