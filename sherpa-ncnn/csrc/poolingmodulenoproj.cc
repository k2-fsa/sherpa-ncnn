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

#include "sherpa-ncnn/csrc/poolingmodulenoproj.h"

namespace sherpa_ncnn {

PoolingModuleNoProj::PoolingModuleNoProj() {
  one_blob_only = false;
  support_inplace = false;
}

int32_t PoolingModuleNoProj::forward(const std::vector<ncnn::Mat> &bottom_blobs,
                                     std::vector<ncnn::Mat> &top_blobs,
                                     const ncnn::Option &opt) const {
  ncnn::Mat x = bottom_blobs[0];
  ncnn::Mat cached_len = bottom_blobs[1];
  ncnn::Mat cached_avg = bottom_blobs[2];

  // x.dims = 2, x.w = C, x.h = T
  // cached_len.dims = 1, cached_len.w = 1
  // cached_avg.dims = 2, cached_avg.w = C, cached_avg.h = 1

  ncnn::Mat &out_x = top_blobs[0];
  out_x.create_like(x, opt.blob_allocator);

  ncnn::Mat &out_cached_len = top_blobs[1];
  out_cached_len.create(cached_len.w, cached_len.elemsize, opt.blob_allocator);

  ncnn::Mat &out_cached_avg = top_blobs[2];
  out_cached_avg.create_like(cached_avg, opt.blob_allocator);

  int32_t w = x.w;
  int32_t h = x.h;

  const float *x_ptr = x;
  const float *cached_avg_ptr = cached_avg;
  float *out_ptr = out_x;

  float n = cached_len[0];

  // process row 0
  for (int32_t c = 0; c < w; ++c) {
    out_ptr[c] = x_ptr[c] + n * cached_avg_ptr[c];
  }

  for (int32_t r = 1; r < h; ++r) {
    const float *x_cur = x.row(r);

    float *out_prev = out_x.row(r - 1);
    float *out_cur = out_x.row(r);

    float scale = 1. / (n + r);  // scale for the previous row
    for (int32_t c = 0; c < w; ++c) {
      out_cur[c] = out_prev[c] + x_cur[c];
      out_prev[c] *= scale;
    }
  }

  float *last_row = out_x.row(h - 1);
  float scale = 1. / (n + h);

  float *out_cached_avg_ptr = out_cached_avg;
  for (int32_t c = 0; c < w; ++c) {
    last_row[c] *= scale;
    out_cached_avg_ptr[c] = last_row[c];
  }

  out_cached_len[0] = n + h;

  return 0;
}

static ncnn::Layer *PoolingModuleNoProjCreator(void * /*userdata*/) {
  return new PoolingModuleNoProj();
}

void RegisterPoolingModuleNoProjLayer(ncnn::Net &net) {
  net.register_custom_layer("PoolingModuleNoProj", PoolingModuleNoProjCreator);
}

}  // namespace sherpa_ncnn
