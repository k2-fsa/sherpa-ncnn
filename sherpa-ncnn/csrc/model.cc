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
#include "sherpa-ncnn/csrc/model.h"

#include <sstream>

#include "sherpa-ncnn/csrc/conv-emformer-model.h"
#include "sherpa-ncnn/csrc/lstm-model.h"
#include "sherpa-ncnn/csrc/meta-data.h"

namespace sherpa_ncnn {

std::string ModelConfig::ToString() const {
  std::ostringstream os;
  os << "encoder_param: " << encoder_param << "\n";
  os << "encoder_bin: " << encoder_bin << "\n";

  os << "decoder_param: " << decoder_param << "\n";
  os << "decoder_bin: " << decoder_bin << "\n";

  os << "joiner_param: " << joiner_param << "\n";
  os << "joiner_bin: " << joiner_bin << "\n";

  os << "num_threads: " << num_threads << "\n";

  return os.str();
}

static bool IsLstmModel(const ncnn::Net &net) {
  for (const auto &layer : net.layers()) {
    if (layer->type == "LSTM") {
      return true;
    }
  }

  return false;
}

static bool IsConvEmformerModel(const ncnn::Net &net) {
  // Note: We may need to add more constraints if number of models gets larger.
  //
  // If the net has a layer of type SherpaMetaData and with name
  // sherpa_meta_data1 and if attribute 0 is 1, we assume the model is
  // a ConvEmformer model

  for (const auto *layer : net.layers()) {
    if (layer->type == "SherpaMetaData" && layer->name == "sherpa_meta_data1") {
      // Note: We don't use dynamic_cast<> here since it will throw
      // the following error
      //  error: ‘dynamic_cast’ not permitted with -fno-rtti
      const auto *meta_data = reinterpret_cast<const MetaData *>(layer);

      if (meta_data->arg0 == 1) return true;
    }
  }

  return false;
}

void Model::InitNet(ncnn::Net &net, const std::string &param,
                    const std::string &bin) {
  if (net.load_param(param.c_str())) {
    NCNN_LOGE("failed to load %s", param.c_str());
    exit(-1);
  }

  if (net.load_model(bin.c_str())) {
    NCNN_LOGE("failed to load %s", bin.c_str());
    exit(-1);
  }
}

std::unique_ptr<Model> Model::Create(const ModelConfig &config) {
  // 1. Load the encoder network
  // 2. If the encoder network has LSTM layers, we assume it is a LstmModel
  // 3. Otherwise, we assume it is a ConvEmformer
  // 4. TODO(fangjun): We need to change this function to support more models
  // in the future

  ncnn::Net net;
  RegisterMetaDataLayer(net);

  auto ret = net.load_param(config.encoder_param.c_str());
  if (ret != 0) {
    NCNN_LOGE("Failed to load %s", config.encoder_param.c_str());
    return nullptr;
  }

  if (IsLstmModel(net)) {
    return std::make_unique<LstmModel>(config);
  }

  if (IsConvEmformerModel(net)) {
    return std::make_unique<ConvEmformerModel>(config);
  }

  return nullptr;
}

}  // namespace sherpa_ncnn
