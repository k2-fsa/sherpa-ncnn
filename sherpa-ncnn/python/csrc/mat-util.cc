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

#include "sherpa-ncnn/python/csrc/mat-util.h"

namespace sherpa_ncnn {

struct KeepMatAlive {
  explicit KeepMatAlive(ncnn::Mat m) : m(m) {}

  ncnn::Mat m;
};

py::array_t<float> MatToArray(ncnn::Mat m) {
  std::vector<py::ssize_t> shape;
  std::vector<py::ssize_t> strides;
  if (m.dims == 1) {
    shape.push_back(m.w);
    strides.push_back(m.elemsize);
  } else if (m.dims == 2) {
    shape.push_back(m.h);
    shape.push_back(m.w);
    strides.push_back(m.w * m.elemsize);
    strides.push_back(m.elemsize);
  } else if (m.dims == 3) {
    shape.push_back(m.c);
    shape.push_back(m.h);
    shape.push_back(m.w);
    strides.push_back(m.cstep * m.elemsize);
    strides.push_back(m.w * m.elemsize);
    strides.push_back(m.elemsize);
  } else if (m.dims == 4) {
    shape.push_back(m.c);
    shape.push_back(m.d);
    shape.push_back(m.h);
    shape.push_back(m.w);
    strides.push_back(m.cstep * m.elemsize);
    strides.push_back(m.w * m.h * m.elemsize);
    strides.push_back(m.w * m.elemsize);
    strides.push_back(m.elemsize);
  }

  auto keep_mat_alive = new KeepMatAlive(m);
  py::capsule handle(keep_mat_alive, [](void *p) {
    delete reinterpret_cast<KeepMatAlive *>(p);
  });

  return py::array_t<float>(shape, strides, (float *)m.data, handle);
}

ncnn::Mat ArrayToMat(py::array array) {
  py::buffer_info info = array.request();
  size_t elemsize = info.itemsize;

  ncnn::Mat ans;

  if (info.ndim == 1) {
    ans = ncnn::Mat((int)info.shape[0], info.ptr, elemsize);
  } else if (info.ndim == 2) {
    ans = ncnn::Mat((int)info.shape[1], (int)info.shape[0], info.ptr, elemsize);
  } else if (info.ndim == 3) {
    ans = ncnn::Mat((int)info.shape[2], (int)info.shape[1], (int)info.shape[0],
                    info.ptr, elemsize);

    // in ncnn, buffer to construct ncnn::Mat need align to ncnn::alignSize
    // with (w * h * elemsize, 16) / elemsize, but the buffer from numpy not
    // so we set the cstep as numpy's cstep
    ans.cstep = (int)info.shape[2] * (int)info.shape[1];
  } else if (info.ndim == 4) {
    ans = ncnn::Mat((int)info.shape[3], (int)info.shape[2], (int)info.shape[1],
                    (int)info.shape[0], info.ptr, elemsize);

    // in ncnn, buffer to construct ncnn::Mat need align to ncnn::alignSize
    // with (w * h * d elemsize, 16) / elemsize, but the buffer from numpy not
    // so we set the cstep as numpy's cstep
    ans.cstep = (int)info.shape[3] * (int)info.shape[2] * (int)info.shape[1];
  }

  return ans;
}

}  // namespace sherpa_ncnn
