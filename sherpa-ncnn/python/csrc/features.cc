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

#include "sherpa-ncnn/python/csrc/features.h"

#include <memory>

#include "sherpa-ncnn/csrc/features.h"

namespace sherpa_ncnn {

void PybindFeatures(py::module *m) {
  using PyClass = FeatureExtractorConfig;
  py::class_<PyClass>(*m, "FeatureExtractorConfig")
      .def(py::init([](int32_t sampling_rate, int32_t feature_dim) {
             auto ans = std::make_unique<PyClass>();
             ans->sampling_rate = sampling_rate;
             ans->feature_dim = feature_dim;
             return ans;
           }),
           py::arg("sampling_rate"), py::arg("feature_dim"))
      .def_readwrite("sampling_rate", &PyClass::sampling_rate)
      .def_readwrite("feature_dim", &PyClass::feature_dim)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_ncnn
