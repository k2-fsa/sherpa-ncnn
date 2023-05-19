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

#ifndef SHERPA_NCNN_CSRC_SIMPLEUPSAMPLE_H_
#define SHERPA_NCNN_CSRC_SIMPLEUPSAMPLE_H_

#include <utility>

#include "layer.h"  // NOLINT
#include "net.h"    // NOLINT

namespace sherpa_ncnn {

class SimpleUpsample : public ncnn::Layer {
 public:
  SimpleUpsample();

  int32_t load_param(const ncnn::ParamDict &pd) override;

  int32_t load_model(const ncnn::ModelBin &mb) override;

  int32_t forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob,
                  const ncnn::Option &opt) const override;

 public:
  int32_t upsample;
  int32_t num_channels;
  int32_t bias_data_size;

  ncnn::Mat bias;
};

void RegisterTensorSimpleUpsampleLayer(ncnn::Net &net);

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_SIMPLEUPSAMPLE_H_
