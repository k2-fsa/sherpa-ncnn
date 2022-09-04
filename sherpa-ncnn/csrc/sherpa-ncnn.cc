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

#include "net.h"
#include <iostream>

static void InitNet(ncnn::Net &net, const std::string &param,
                    const std::string &model) {
  if (net.load_param(param.c_str())) {
    std::cerr << "failed to load " << param << "\n";
    exit(-1);
  }

  if (net.load_model(model.c_str())) {
    std::cerr << "failed to load " << model << "\n";
    exit(-1);
  }
}

int main() {

  std::string encoder_param =
      "bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param";

  std::string encoder_model =
      "bar/encoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin";

  std::string decoder_param =
      "bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.param";

  std::string decoder_model =
      "bar/decoder_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin";

  std::string joiner_param =
      "bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.param";

  std::string joiner_model =
      "bar/joiner_jit_trace-iter-468000-avg-16-pnnx.ncnn.bin";

  ncnn::Net encoder_net;
  ncnn::Net decoder_net;
  ncnn::Net joiner_net;

  InitNet(encoder_net, encoder_param, encoder_model);
  InitNet(decoder_net, decoder_param, decoder_model);
  InitNet(joiner_net, joiner_param, joiner_model);
}
