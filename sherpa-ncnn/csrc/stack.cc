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

#include "sherpa-ncnn/csrc/stack.h"

namespace sherpa_ncnn {

Stack::Stack() {
  one_blob_only = false;
  support_inplace = false;
}

int32_t Stack::load_param(const ncnn::ParamDict &pd) {
  axis = pd.get(0, 0);
  if (axis != 0) {
    NCNN_LOGE("Stack: Only axis==0 is implemented. Given %d", axis);
    return -100;
  }

  return 0;
}

int32_t Stack::forward(const std::vector<ncnn::Mat> &bottom_blobs,
                       std::vector<ncnn::Mat> &top_blobs,
                       const ncnn::Option &opt) const {
  int32_t dims = bottom_blobs[0].dims;
  size_t elemsize = bottom_blobs[0].elemsize;

  if (dims == 1) {
    int32_t out_w = bottom_blobs[0].w;
    int32_t out_h = bottom_blobs.size();

    ncnn::Mat &top_blob = top_blobs[0];
    top_blob.create(out_w, out_h, elemsize, opt.blob_allocator);
    if (top_blob.empty()) return -100;

    unsigned char *outptr = top_blob;

    size_t bytes_per_blob = out_w * elemsize;

    for (size_t b = 0; b < bottom_blobs.size(); ++b) {
      const unsigned char *ptr = bottom_blobs[b];
      memcpy(outptr, ptr, bytes_per_blob);

      outptr += bytes_per_blob;
    }

    return 0;
  }

  if (dims == 2) {
    int32_t out_w = bottom_blobs[0].w;
    int32_t out_h = bottom_blobs[0].h;
    int32_t out_c = bottom_blobs.size();

    ncnn::Mat &top_blob = top_blobs[0];
    top_blob.create(out_w, out_h, out_c, elemsize, opt.blob_allocator);
    if (top_blob.empty()) return -100;

    size_t bytes_per_blob = out_w * out_h * elemsize;

    for (size_t b = 0; b < bottom_blobs.size(); ++b) {
      unsigned char *outptr = top_blob.channel(b);
      const unsigned char *ptr = bottom_blobs[b];

      memcpy(outptr, ptr, bytes_per_blob);
    }

    return 0;
  }

  NCNN_LOGE("Stack: dim %d is not implemented", dims);

  return -100;
}

static ncnn::Layer *StackCreator(void * /*userdata*/) { return new Stack(); }

void RegisterStackLayer(ncnn::Net &net) {
  net.register_custom_layer("Stack", StackCreator);
}

}  // namespace sherpa_ncnn
