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

#include "sherpa-ncnn/csrc/simpleupsample.h"

namespace sherpa_ncnn {

SimpleUpsample::SimpleUpsample() {
  one_blob_only = true;
  support_inplace = false;
}

int32_t SimpleUpsample::load_param(const ncnn::ParamDict &pd) {
  upsample = pd.get(0, 0);
  num_channels = pd.get(1, 0);
  bias_data_size = pd.get(2, 0);
  if (bias_data_size != upsample * num_channels) {
    NCNN_LOGE("upsample: %d, num_channels: %d, bias_data_size: %d. %dx%d!=%d",
              upsample, num_channels, bias_data_size, upsample, num_channels,
              bias_data_size);
    return -100;
  }

  return 0;
}

int32_t SimpleUpsample::load_model(const ncnn::ModelBin &mb) {
  bias = mb.load(num_channels, upsample, 0);
  if (bias.empty()) return -100;

  return 0;
}

int32_t SimpleUpsample::forward(const ncnn::Mat &bottom_blob,
                                ncnn::Mat &top_blob,
                                const ncnn::Option &opt) const {
  // bottom_blob.dims == 2
  // bottom_blob.w == seq_len
  // bottom_blob.h == num_channels

  int32_t outw = bottom_blob.w;
  int32_t outh = upsample;
  int32_t outc = bottom_blob.h;
  size_t elemsize = bottom_blob.elemsize;

  top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
  if (top_blob.empty()) return -100;

#pragma omp parallel for num_threads(opt.num_threads)
  for (int32_t q = 0; q < outc; ++q) {
    ncnn::Mat out_m = top_blob.channel(q);
    const float *a_ptr = bottom_blob.row(q);

    for (int32_t y = 0; y < outh; ++y) {
      float *out_ptr = out_m.row(y);
      const float *b_ptr = bias.row(y);
      for (int32_t x = 0; x < outw; ++x) {
        out_ptr[x] = a_ptr[x] + b_ptr[x];
      }
    }
  }

  top_blob = top_blob.reshape(outw, outh * outc);

  return 0;
}

static ncnn::Layer *SimpleUpsampleCreator(void * /*userdata*/) {
  return new SimpleUpsample();
}

void RegisterTensorSimpleUpsampleLayer(ncnn::Net &net) {
  net.register_custom_layer("SimpleUpsample", SimpleUpsampleCreator);
}

}  // namespace sherpa_ncnn
