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

#include "net.h"
namespace sherpa_ncnn {

ncnn::Layer *MetaDataCreator(void * /*userdata*/) { return new MetaData(); }

/*
In encoder.ncnn.param, you can use

SherpaMetaData sherpa_meta_data1 0 0 0=1 1=12 24=1.5
 */

void RegisterMetaDataLayer(ncnn::Net &net) {
  net.register_custom_layer("SherpaMetaData", MetaDataCreator);
}

}  // namespace sherpa_ncnn
