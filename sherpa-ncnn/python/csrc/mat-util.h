/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#ifndef SHERPA_NCNN_PYTHON_CSRC_MAT_UTIL_H_
#define SHERPA_NCNN_PYTHON_CSRC_MAT_UTIL_H_

#include "mat.h"
#include "sherpa-ncnn/python/csrc/sherpa-ncnn.h"

namespace sherpa_ncnn {

// Convert a ncnn::Mat to a numpy array. Data is shared.
//
// @param m It should be a float unpacked matrix
py::array_t<float> MatToArray(ncnn::Mat m);

// convert an array to a ncnn::Mat
ncnn::Mat ArrayToMat(py::array array);

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_PYTHON_CSRC_MODEL_UTIL_H_
