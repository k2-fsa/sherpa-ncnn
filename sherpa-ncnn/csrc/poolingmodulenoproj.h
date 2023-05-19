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
#ifndef SHERPA_NCNN_CSRC_POOLINGMODULENOPROJ_H_
#define SHERPA_NCNN_CSRC_POOLINGMODULENOPROJ_H_

#include <utility>
#include <vector>

#include "layer.h"  // NOLINT
#include "net.h"    // NOLINT

namespace sherpa_ncnn {

// used only by zipformer
class PoolingModuleNoProj : public ncnn::Layer {
 public:
  PoolingModuleNoProj();

  int32_t forward(const std::vector<ncnn::Mat> &bottom_blobs,
                  std::vector<ncnn::Mat> &top_blobs,
                  const ncnn::Option &opt) const override;
};

void RegisterPoolingModuleNoProjLayer(ncnn::Net &net);

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_POOLINGMODULENOPROJ_H_
