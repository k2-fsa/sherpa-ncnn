// sherpa-ncnn/python/csrc/offline-stream.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/python/csrc/offline-stream.h"

#include <vector>

#include "sherpa-ncnn/csrc/offline-stream.h"

namespace sherpa_ncnn {

constexpr const char *kAcceptWaveformUsage = R"(
Process audio samples.

Args:
  sample_rate:
    Sample rate of the input samples. If it is different from the one
    expected by the model, we will do resampling inside.
  waveform:
    A 1-D float32 tensor containing audio samples. It must be normalized
    to the range [-1, 1].
)";

static void PybindOfflineRecognizerResult(py::module *m) {  // NOLINT
  using PyClass = OfflineRecognizerResult;
  py::class_<PyClass>(*m, "OfflineRecognizerResult")
      .def("__str__", &PyClass::AsJsonString)
      .def_property_readonly(
          "text",
          [](const PyClass &self) -> py::str {
            return py::str(PyUnicode_DecodeUTF8(self.text.c_str(),
                                                self.text.size(), "ignore"));
          })
      .def_property_readonly("lang",
                             [](const PyClass &self) { return self.lang; })
      .def_property_readonly("emotion",
                             [](const PyClass &self) { return self.emotion; })
      .def_property_readonly("event",
                             [](const PyClass &self) { return self.event; })
      .def_property_readonly("tokens",
                             [](const PyClass &self) { return self.tokens; })
      .def_property_readonly(
          "timestamps", [](const PyClass &self) { return self.timestamps; });
}

void PybindOfflineStream(py::module *m) {
  PybindOfflineRecognizerResult(m);

  using PyClass = OfflineStream;
  py::class_<PyClass>(*m, "OfflineStream")
      .def(
          "accept_waveform",
          [](PyClass &self, float sample_rate,
             const std::vector<float> &waveform) {
            self.AcceptWaveform(sample_rate, waveform.data(), waveform.size());
          },
          py::arg("sample_rate"), py::arg("waveform"), kAcceptWaveformUsage,
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("result", &PyClass::GetResult);
}

}  // namespace sherpa_ncnn
