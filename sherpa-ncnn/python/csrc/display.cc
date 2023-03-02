
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

#include "sherpa-ncnn/python/csrc/display.h"

#include "sherpa-ncnn/csrc/display.h"

namespace sherpa_ncnn {

void PybindDisplay(py::module *m) {
  using PyClass = Display;
  py::class_<PyClass>(*m, "Display")
      .def(py::init<int32_t>(), py::arg("max_word_per_line") = 60)
      .def("print", &PyClass::Print, py::arg("idx"), py::arg("s"));
}

}  // namespace sherpa_ncnn
