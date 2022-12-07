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
  int load_param(const ncnn::ParamDict &pd) override {
    arg0 = pd.get(0, 0), arg1 = pd.get(1, 0), arg2 = pd.get(2, 0);
    arg3 = pd.get(3, 0), arg4 = pd.get(4, 0), arg5 = pd.get(5, 0);
    arg6 = pd.get(6, 0), arg7 = pd.get(7, 0), arg8 = pd.get(8, 0);
    arg9 = pd.get(9, 0), arg10 = pd.get(10, 0), arg11 = pd.get(11, 0);
    arg12 = pd.get(12, 0), arg13 = pd.get(13, 0), arg14 = pd.get(14, 0);
    arg15 = pd.get(15, 0), arg16 = pd.get(16, 0), arg17 = pd.get(17, 0);
    arg18 = pd.get(18, 0), arg19 = pd.get(19, 0), arg20 = pd.get(20, 0);
    arg21 = pd.get(21, 0), arg22 = pd.get(22, 0), arg23 = pd.get(23, 0);

    // The following 8 attributes are of type float
    arg24 = pd.get(24, 0.f), arg25 = pd.get(25, 0.f), arg26 = pd.get(26, 0.f);
    arg27 = pd.get(27, 0.f), arg28 = pd.get(28, 0.f), arg29 = pd.get(29, 0.f);
    arg30 = pd.get(30, 0.f), arg31 = pd.get(31, 0.f);

    return 0;
  }

  int32_t arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7;
  int32_t arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15;
  int32_t arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23;

  float arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31;
};

ncnn::Layer *MetaDataCreator(void *userdata);

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
