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

#include "sherpa-ncnn/csrc/tensorasstrided.h"

namespace sherpa_ncnn {

TensorAsStrided::TensorAsStrided() {
  one_blob_only = true;
  support_inplace = false;
}

int32_t TensorAsStrided::load_param(const ncnn::ParamDict &pd) {
  sizes = pd.get(0, ncnn::Mat());
  strides = pd.get(1, ncnn::Mat());
  storage_offset = pd.get(2, 0);

  if (sizes.dims != 1 && strides.dims != 1) {
    if (sizes.dims != 0) {
      NCNN_LOGE("sizes.dims: %d, strides.dims: %d. They are not 1!\n",
                sizes.dims, strides.dims);
      return -100;
    }
  }

  if (sizes.w != strides.w) {
    NCNN_LOGE("sizes.w: %d, strides.w: %d. They are not equal!\n", sizes.w,
              strides.w);
    return -100;
  }

  return 0;
}

int32_t TensorAsStrided::forward(const ncnn::Mat &bottom_blob,
                                 ncnn::Mat &top_blob,
                                 const ncnn::Option &opt) const {
  const int32_t *p_sizes = sizes;
  const int32_t *p_strides = strides;

  if (sizes.w == 3) {
    if (bottom_blob.dims != 3) {
      NCNN_LOGE("Only 3-D tensors are supported right now");
      return -100;
    }

    int32_t inc = bottom_blob.c;
    int32_t inh = bottom_blob.h;
    int32_t inw = bottom_blob.w;

    int32_t outc = p_sizes[0];
    int32_t outh = p_sizes[1];
    int32_t outw = p_sizes[2];

    if (bottom_blob.c != outc) {
      NCNN_LOGE("We only implement in_c == out_c right now");
      return -100;
    }

    if (p_strides[0] != inh * inw) {
      NCNN_LOGE("Stride that crosses channels is not supported");
      return -100;
    }

    size_t elemsize = bottom_blob.elemsize;
    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);

    int32_t stride1 = p_strides[1];
    int32_t stride2 = p_strides[2];

#pragma omp parallel for num_threads(opt.num_threads)
    for (int32_t q = 0; q < outc; q++) {
      ncnn::Mat out_m = top_blob.channel(q);

      const float *in_m = bottom_blob.channel(q);
      in_m += storage_offset;

      for (int32_t y = 0; y < outh; ++y) {
        float *out_ptr = out_m.row(y);
        const float *in_ptr = in_m + y * stride1;
        for (int32_t x = 0; x < outw; ++x) {
          out_ptr[x] = in_ptr[x * stride2];
        }
      }
    }

    return 0;
  }

  NCNN_LOGE("TensorAsStrided: Only 3-D tensors are supported right now");

  return -100;
}

static ncnn::Layer *TensorAsStridedCreator(void * /*userdata*/) {
  return new TensorAsStrided();
}

void RegisterTensorAsStridedLayer(ncnn::Net &net) {
  net.register_custom_layer("TensorAsStrided", TensorAsStridedCreator);
}

}  // namespace sherpa_ncnn
