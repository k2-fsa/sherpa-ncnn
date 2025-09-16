// sherpa-ncnn/python/csrc/offline-tts.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa-ncnn/python/csrc/offline-tts.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "sherpa-ncnn/csrc/offline-tts.h"
#include "sherpa-ncnn/python/csrc/offline-tts-model-config.h"

namespace sherpa_ncnn {

static void PybindGeneratedAudio(py::module *m) {
  using PyClass = GeneratedAudio;
  py::class_<PyClass>(*m, "GeneratedAudio")
      .def(py::init<>())
      .def_readwrite("samples", &PyClass::samples)
      .def_readwrite("sample_rate", &PyClass::sample_rate)
      .def("__str__", [](PyClass &self) {
        std::ostringstream os;
        os << "GeneratedAudio(sample_rate=" << self.sample_rate << ", ";
        os << "num_samples=" << self.samples.size() << ")";
        return os.str();
      });
}

static void PybindTtsArgs(py::module *m) {
  using PyClass = TtsArgs;
  py::class_<PyClass>(*m, "TtsArgs")
      .def(py::init<const std::string &,
                    const std::vector<std::vector<int32_t>> &, int32_t, float,
                    float, float>(),
           py::arg("text") = "",
           py::arg("tokens") = std::vector<std::vector<int32_t>>{},
           py::arg("sid") = 0, py::arg("speed") = 1.0f,
           py::arg("noise_scale") = 0.667f, py::arg("noise_scale_w") = 0.8f)
      .def_readwrite("text", &PyClass::text)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("sid", &PyClass::sid)
      .def_readwrite("speed", &PyClass::speed)
      .def_readwrite("noise_scale", &PyClass::noise_scale)
      .def_readwrite("noise_scale_w", &PyClass::noise_scale_w)
      .def("__str__", [](PyClass &self) {
        std::ostringstream os;
        os << "TtsArgs(";
        os << "text=\"" << self.text << "\"";
        os << ", tokens=";
        if (self.tokens.empty()) {
          os << "[]";
        } else {
          os << "[";
          std::string sep;
          for (const auto &ids : self.tokens) {
            os << sep << "[";
            sep = "";
            for (int32_t i : ids) {
              os << sep << i;
              sep = ", ";
            }
            os << "]";
            sep = ", ";
          }
          os << "]";
        }

        os << ", sid=" << self.sid;
        os << ", speed=" << self.speed;
        os << ", noise_scale=" << self.noise_scale;
        os << ", noise_scale_w=" << self.noise_scale_w;
        os << ")";
        return os.str();
      });
}

static void PybindOfflineTtsConfig(py::module *m) {
  PybindOfflineTtsModelConfig(m);

  using PyClass = OfflineTtsConfig;
  py::class_<PyClass>(*m, "OfflineTtsConfig")
      .def(py::init<>())
      .def(py::init<const OfflineTtsModelConfig &, const std::string &,
                    const std::string &, int32_t, float>(),
           py::arg("model"), py::arg("rule_fsts") = "",
           py::arg("rule_fars") = "", py::arg("max_num_sentences") = 1,
           py::arg("silence_scale") = 0.2)
      .def_readwrite("model", &PyClass::model)
      .def_readwrite("rule_fsts", &PyClass::rule_fsts)
      .def_readwrite("rule_fars", &PyClass::rule_fars)
      .def_readwrite("max_num_sentences", &PyClass::max_num_sentences)
      .def_readwrite("silence_scale", &PyClass::silence_scale)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineTts(py::module *m) {
  PybindOfflineTtsConfig(m);
  PybindGeneratedAudio(m);
  PybindTtsArgs(m);

  using PyClass = OfflineTts;
  py::class_<PyClass>(*m, "OfflineTts")
      .def(py::init<const OfflineTtsConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("sample_rate", &PyClass::SampleRate)
      .def_property_readonly("num_speakers", &PyClass::NumSpeakers)
      .def(
          "generate",
          [](const PyClass &self, const TtsArgs &args,
             std::function<int32_t(py::array_t<float>, int32_t, int32_t)>
                 callback) -> GeneratedAudio {
            if (!callback) {
              return self.Generate(args);
            }

            std::function<int32_t(const float *, int32_t, int32_t, int32_t,
                                  void *)>
                callback_wrapper = [callback](const float *samples, int32_t n,
                                              int32_t processed, int32_t total,
                                              void *) {
                  // CAUTION(fangjun): we have to copy samples since it is
                  // freed once the call back returns.

                  pybind11::gil_scoped_acquire acquire;

                  pybind11::array_t<float> array(n);
                  py::buffer_info buf = array.request();
                  auto p = static_cast<float *>(buf.ptr);
                  std::copy(samples, samples + n, p);
                  return callback(array, processed, total);
                };

            return self.Generate(args, callback_wrapper);
          },
          py::arg("args"), py::arg("callback") = py::none(),
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_ncnn
