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

#include "sherpa-ncnn/python/csrc/decoder.h"

#include <string>

#include "sherpa-ncnn//csrc/decoder.h"

namespace sherpa_ncnn {

void PybindDecoder(py::module *m) {
  using PyClass = DecoderConfig;
  py::class_<PyClass>(*m, "DecoderConfig")
      .def(py::init<const std::string &, int32_t>(), py::arg("method"),
           py::arg("num_active_paths"))
      .def_readwrite("method", &PyClass::method)
      .def_readwrite("num_active_paths", &PyClass::num_active_paths)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_ncnn
