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

#include "sherpa-ncnn/csrc/meta-data.h"

#include "net.h"  // NOLINT
namespace sherpa_ncnn {

int MetaData::load_param(const ncnn::ParamDict &pd) {
  arg0 = pd.get(0, 0);
  arg1 = pd.get(1, 0), arg2 = pd.get(2, 0), arg3 = pd.get(3, 0);
  arg4 = pd.get(4, 0), arg5 = pd.get(5, 0), arg6 = pd.get(6, 0);
  arg7 = pd.get(7, 0), arg8 = pd.get(8, 0), arg9 = pd.get(9, 0);
  arg10 = pd.get(10, 0), arg11 = pd.get(11, 0), arg12 = pd.get(12, 0);
  arg13 = pd.get(13, 0), arg14 = pd.get(14, 0), arg15 = pd.get(15, 0);

  arg16 = pd.get(16, ncnn::Mat()), arg17 = pd.get(17, ncnn::Mat());
  arg18 = pd.get(18, ncnn::Mat()), arg19 = pd.get(19, ncnn::Mat());
  arg20 = pd.get(20, ncnn::Mat()), arg21 = pd.get(21, ncnn::Mat());
  arg22 = pd.get(22, ncnn::Mat()), arg23 = pd.get(23, ncnn::Mat());

  // The following 8 attributes are of type float
  arg24 = pd.get(24, 0.f), arg25 = pd.get(25, 0.f), arg26 = pd.get(26, 0.f);
  arg27 = pd.get(27, 0.f), arg28 = pd.get(28, 0.f), arg29 = pd.get(29, 0.f);
  arg30 = pd.get(30, 0.f), arg31 = pd.get(31, 0.f);

  return 0;
}

static ncnn::Layer *MetaDataCreator(void * /*userdata*/) {
  return new MetaData();
}

/*
In encoder.ncnn.param, you can use

SherpaMetaData sherpa_meta_data1 0 0 0=1 1=12 24=1.5
 */

void RegisterMetaDataLayer(ncnn::Net &net) {
  net.register_custom_layer("SherpaMetaData", MetaDataCreator);
}

}  // namespace sherpa_ncnn
