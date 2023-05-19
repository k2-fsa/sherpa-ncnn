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
#include "sherpa-ncnn/csrc/poolingmodulenoproj.h"
#include "sherpa-ncnn/csrc/simpleupsample.h"
#include "sherpa-ncnn/csrc/stack.h"
#include "sherpa-ncnn/csrc/tensorasstrided.h"
#include "sherpa-ncnn/csrc/zipformer-model.h"

namespace sherpa_ncnn {

std::string ModelConfig::ToString() const {
  std::ostringstream os;
  os << "ModelConfig(";
  os << "encoder_param=\"" << encoder_param << "\", ";
  os << "encoder_bin=\"" << encoder_bin << "\", ";
  os << "decoder_param=\"" << decoder_param << "\", ";
  os << "decoder_bin=\"" << decoder_bin << "\", ";
  os << "joiner_param=\"" << joiner_param << "\", ";
  os << "joiner_bin=\"" << joiner_bin << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "encoder num_threads=" << encoder_opt.num_threads << ", ";
  os << "decoder num_threads=" << decoder_opt.num_threads << ", ";
  os << "joiner num_threads=" << joiner_opt.num_threads << ")";

  return os.str();
}

static bool IsLstmModel(const ncnn::Net &net) {
  for (const auto *layer : net.layers()) {
    if (layer->type == "SherpaMetaData" && layer->name == "sherpa_meta_data1") {
      // Note: We don't use dynamic_cast<> here since it will throw
      // the following error
      //  error: ‘dynamic_cast’ not permitted with -fno-rtti
      const auto *meta_data = reinterpret_cast<const MetaData *>(layer);

      if (meta_data->arg0 == 3) return true;
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

static bool IsZipformerModel(const ncnn::Net &net) {
  // Note: We may need to add more constraints if number of models gets larger.
  //
  // If the net has a layer of type SherpaMetaData and with name
  // sherpa_meta_data1 and if attribute 0 is 2, we assume the model is
  // a Zipformer model.

  for (const auto *layer : net.layers()) {
    if (layer->type == "SherpaMetaData" && layer->name == "sherpa_meta_data1") {
      // Note: We don't use dynamic_cast<> here since it will throw
      // the following error
      //  error: ‘dynamic_cast’ not permitted with -fno-rtti
      const auto *meta_data = reinterpret_cast<const MetaData *>(layer);

      if (meta_data->arg0 == 2) {
        // arg15 is the version.
        // Staring from sherpa-ncnn 2.0, we use the master of tencent/ncnn
        // directly and we have update the version of Zipformer from 0 to 1.
        //
        // If yo are using an older version of Zipformer, please
        // re-download the model or re-export the model using the latest icefall
        // or use sherpa-ncnn < v2.0
        if (meta_data->arg15 < 1) {
          NCNN_LOGE(
              "You are using a too old version of Zipformer. You can "
              "choose one of the following solutions: \n"
              "  (1) Re-download the latest model\n"
              "  (2) Re-export your model using the latest icefall. Remember "
              "to strictly follow the documentation\n"
              "      to update the version number to 1.\n"
              "  (3) Use sherpa-ncnn < v2.0 (not recommended)\n");
          exit(-1);
        }
        return true;
      }
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

#if __ANDROID_API__ >= 9
void Model::InitNet(AAssetManager *mgr, ncnn::Net &net,
                    const std::string &param, const std::string &bin) {
  if (net.load_param(mgr, param.c_str())) {
    NCNN_LOGE("failed to load %s", param.c_str());
    exit(-1);
  }

  if (net.load_model(mgr, bin.c_str())) {
    NCNN_LOGE("failed to load %s", bin.c_str());
    exit(-1);
  }
}
#endif

void Model::RegisterCustomLayers(ncnn::Net &net) {
  RegisterMetaDataLayer(net);

  RegisterPoolingModuleNoProjLayer(net);   // for zipformer only
  RegisterTensorAsStridedLayer(net);       // for zipformer only
  RegisterTensorSimpleUpsampleLayer(net);  // for zipformer only
  RegisterStackLayer(net);                 // for zipformer only
}

std::unique_ptr<Model> Model::Create(const ModelConfig &config) {
  // 1. Load the encoder network
  // 2. If the encoder network has LSTM layers, we assume it is a LstmModel
  // 3. Otherwise, we assume it is a ConvEmformer
  // 4. TODO(fangjun): We need to change this function to support more models
  // in the future

  ncnn::Net net;
  RegisterCustomLayers(net);

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

  if (IsZipformerModel(net)) {
    return std::make_unique<ZipformerModel>(config);
  }

  NCNN_LOGE(
      "Unable to create a model from specified model files.\n"
      "Please check: \n"
      "  1. If you are using a ConvEmformer/Zipformer/LSTM model, please "
      "make "
      "sure "
      "you have added SherapMetaData to encoder_xxx.ncnn.param "
      "(or encoder_xxx.ncnn.int8.param if you are using an int8 model). "
      "You need to add it manually after converting the model with pnnx.\n"
      "  2. (Android) Whether the app requires an int8 model or not\n");

  return nullptr;
}

#if __ANDROID_API__ >= 9
std::unique_ptr<Model> Model::Create(AAssetManager *mgr,
                                     const ModelConfig &config) {
  ncnn::Net net;
  RegisterCustomLayers(net);

  auto ret = net.load_param(mgr, config.encoder_param.c_str());
  if (ret != 0) {
    NCNN_LOGE("Failed to load %s", config.encoder_param.c_str());
    return nullptr;
  }

  if (IsLstmModel(net)) {
    return std::make_unique<LstmModel>(mgr, config);
  }

  if (IsConvEmformerModel(net)) {
    return std::make_unique<ConvEmformerModel>(mgr, config);
  }

  if (IsZipformerModel(net)) {
    return std::make_unique<ZipformerModel>(mgr, config);
  }

  NCNN_LOGE(
      "Unable to create a model from specified model files.\n"
      "Please check: \n"
      "  1. If you are using a ConvEmformer/Zipformer/LSTM model, please "
      "make "
      "sure "
      "you have added SherapMetaData to encoder_xxx.ncnn.param "
      "(or encoder_xxx.ncnn.int8.param if you are using an int8 model). "
      "You need to add it manually after converting the model with pnnx.\n"
      "  2. (Android) Whether the app requires an int8 model or not\n");

  return nullptr;
}
#endif

}  // namespace sherpa_ncnn
