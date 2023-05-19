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

#ifndef SHERPA_NCNN_CSRC_META_DATA_H_
#define SHERPA_NCNN_CSRC_META_DATA_H_

#include "layer.h"  // NOLINT
#include "net.h"    // NOLINT

namespace sherpa_ncnn {

class MetaData : public ncnn::Layer {
 public:
  int load_param(const ncnn::ParamDict &pd) override;

  // arg0 is the model type:
  //  1 - ConvEmformer
  //  2 - Zipformer
  //  3 - LSTM
  //
  //  arg15 is the model version, defaults to 0
  int32_t arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7;
  int32_t arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15;

  ncnn::Mat arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23;

  float arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31;
};

/*
In encoder.ncnn.param, you can use

SherpaMetaData sherpa_meta_data1 0 0 0=1 1=12 24=1.5

For instace, suppose you have a encoder.ncnn.param looks like below:


7767517
1060 1342
Input                    in0                      0 1 in0

You can change it to

7767517
1061 1342
SherpaMetaData            sherpa_meta_data1       0 0 0=1
Input                    in0                      0 1 in0

Note: You first need to change 1060 to 1061 since we add one layer

 */

void RegisterMetaDataLayer(ncnn::Net &net);

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_META_DATA_H_
