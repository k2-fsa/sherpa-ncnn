/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 * Copyright (c)  2022                     (Pingfeng Luo)
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

#include "sherpa-ncnn/csrc/recognizer.h"
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-ncnn/csrc/decoder.h"
#include "sherpa-ncnn/csrc/greedy-search-decoder.h"
#include "sherpa-ncnn/csrc/modified-beam-search-decoder.h"

namespace sherpa_ncnn {

static RecognitionResult Convert(const DecoderResult &src,
                                 const SymbolTable &sym_table,
                                 int32_t frame_shift_ms,
                                 int32_t subsampling_factor) {
  RecognitionResult ans;
  ans.stokens.reserve(src.tokens.size());
  ans.timestamps.reserve(src.timestamps.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];
    text.append(sym);
    ans.stokens.push_back(sym);
  }

  ans.text = std::move(text);
  ans.tokens = src.tokens;
  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    ans.timestamps.push_back(time);
  }
  return ans;
}

std::string RecognitionResult::ToString() const {
  std::ostringstream os;

  os << "text: " << text << "\n";
  os << "timestamps: ";
  for (const auto &t : timestamps) {
    os << t << " ";
  }
  os << "\n";

  return os.str();
}

std::string RecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "RecognizerConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "decoder_config=" << decoder_config.ToString() << ", ";
  os << "max_active_paths=" << max_active_paths << ", ";
  os << "endpoint_config=" << endpoint_config.ToString() << ", ";
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ", ";
  os << "context_score=" << context_score << ", ";
  os << "decoding_method=\"" << decoding_method << "\")";

  return os.str();
}

class Recognizer::Impl {
 public:
  explicit Impl(const RecognizerConfig &config)
      : config_(config),
        model_(Model::Create(config.model_config)),
        endpoint_(config.endpoint_config),
        sym_(config.model_config.tokens) {
    if (config.decoder_config.method == "greedy_search") {
      decoder_ = std::make_unique<GreedySearchDecoder>(model_.get());
    } else if (config.decoder_config.method == "modified_beam_search") {
      decoder_ = std::make_unique<ModifiedBeamSearchDecoder>(
          model_.get(), config.decoder_config.num_active_paths);
    } else {
      NCNN_LOGE("Unsupported method: %s", config.decoder_config.method.c_str());
      exit(-1);
    }
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const RecognizerConfig &config)
      : config_(config),
        model_(Model::Create(mgr, config.model_config)),
        endpoint_(config.endpoint_config),
        sym_(mgr, config.model_config.tokens) {
    if (config.decoder_config.method == "greedy_search") {
      decoder_ = std::make_unique<GreedySearchDecoder>(model_.get());
    } else if (config.decoder_config.method == "modified_beam_search") {
      decoder_ = std::make_unique<ModifiedBeamSearchDecoder>(
          model_.get(), config.decoder_config.num_active_paths);
    } else {
      NCNN_LOGE("Unsupported method: %s", config.decoder_config.method.c_str());
      exit(-1);
    }
  }
#endif

  std::unique_ptr<Stream> CreateStream() const {
    auto stream = std::make_unique<Stream>(config_.feat_config);
    stream->SetResult(decoder_->GetEmptyResult());
    stream->SetStates(model_->GetEncoderInitStates());
    return stream;
  }

  // std::unique_ptr<Stream> CreateStream(const std::vector<std::vector<int32_t>>& contexts) const {
  std::unique_ptr<Stream> CreateStream(const char* contexts) const {
    std::vector<std::vector<int32_t>> hotwords;
    std::vector<int32_t> tmp;
    // std::unordered_map<std::string, int> dictionary;
    // std::ifstream file(config_.model_config.tokens);
    // if (!file) {
    //     std::cerr << "open file failed: " << config_.model_config.tokens << std::endl;
    //     return nullptr;
    // }
    // std::string line;
    // while (std::getline(file, line)) {
    //   std::istringstream iss(line);
    //   std::string first;
    //   std::string second;
    //   int number = 0;
    //   std::size_t spacePos = line.find(' ');
    //   if (spacePos != std::string::npos) {
    //       first = line.substr(0, spacePos);
    //       second = line.substr(spacePos + 1);
    //   } else {
    //       std::cerr<<"token is error"<<std::endl;
    //       first = line;
    //       second = "";
		//   return nullptr;
    //   }
    //   dictionary[first] = std::stoi(second);
    // }
    // file.close();
	  /*each line in hotwords file is a string which is segmented by space*/
	  std::string hotwordsfile(contexts);
    std::ifstream file1(hotwordsfile);
    if (!file1) {
      std::cerr << "open file failed: " << hotwordsfile << std::endl;
      return nullptr;
    }
    std::string lines;
	  std::string word;
    while (std::getline(file1, lines)) {
      std::istringstream iss(lines);
	    while(iss >> word){
	      std::cout<<word<<" ";
        if (sym2id_.find(word) != sym2id_.end()) {
          int number = sym2id_[word];
          tmp.push_back(number);
        } else {
          std::cout << "can't find id" << std::endl;
		      return nullptr;
        }
      //std::cout<<std::endl;
      }
      hotwords.push_back(tmp);
      tmp.clear();
    }
    auto r = decoder_->GetEmptyResult();
    auto context_graph =
        std::make_shared<ContextGraph>(hotwords, config_.context_score);
    auto stream =
        std::make_unique<Stream>(config_.feat_config, context_graph);
    if (config_.decoder_config.method == "modified_beam_search" &&
        nullptr != stream->GetContextGraph()) {
      std::cout<<"create contexts stream"<<std::endl;
      // r.hyps has only one element.
      for (auto it = r.hyps.begin(); it != r.hyps.end(); ++it) {
        it->second.context_state = stream->GetContextGraph()->Root();
      }
    }
    stream->SetResult(r);
    stream->SetStates(model_->GetEncoderInitStates());
    return stream;
  }

  bool IsReady(Stream *s) const {
    return s->GetNumProcessedFrames() + model_->Segment() < s->NumFramesReady();
  }

  void DecodeStream(Stream *s) const {
    int32_t segment = model_->Segment();
    int32_t offset = model_->Offset();
    bool has_context_graph = false;

    if (!has_context_graph && s->GetContextGraph())
      has_context_graph = true;

    ncnn::Mat features = s->GetFrames(s->GetNumProcessedFrames(), segment);
    s->GetNumProcessedFrames() += offset;
    std::vector<ncnn::Mat> states = s->GetStates();

    ncnn::Mat encoder_out;
    std::tie(encoder_out, states) = model_->RunEncoder(features, states);

//    decoder_->Decode(encoder_out, &s->GetResult());
    if (has_context_graph) {
      decoder_->Decode(encoder_out, s, &s->GetResult());
    } else {
      decoder_->Decode(encoder_out, &s->GetResult());
    }
    s->SetStates(states);
  }

  bool IsEndpoint(Stream *s) const {
    if (!config_.enable_endpoint) return false;
    int32_t num_processed_frames = s->GetNumProcessedFrames();

    // frame shift is 10 milliseconds
    float frame_shift_in_seconds = 0.01;

    // subsampling factor is 4
    int32_t trailing_silence_frames = s->GetResult().num_trailing_blanks * 4;

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(Stream *s) const {
    auto r = decoder_->GetEmptyResult();
    if (config_.decoding_method == "modified_beam_search" &&
        nullptr != s->GetContextGraph()) {
      for (auto it = r.hyps.begin(); it != r.hyps.end(); ++it) {
        it->second.context_state = s->GetContextGraph()->Root();
      }
    }
    // Caution: We need to keep the decoder output state
    ncnn::Mat decoder_out = s->GetResult().decoder_out;
    s->SetResult(decoder_->GetEmptyResult());
    s->GetResult().decoder_out = decoder_out;

    // don't reset encoder state
    // s->SetStates(model_->GetEncoderInitStates());

    // reset feature extractor
    // Note: We only reset the counter. The underlying audio samples are
    // still kept in memory
    s->Reset();
  }

  RecognitionResult GetResult(Stream *s) const {
    DecoderResult decoder_result = s->GetResult();
    decoder_->StripLeadingBlanks(&decoder_result);

    // Those 2 parameters are figured out from sherpa source code
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    return Convert(decoder_result, sym_, frame_shift_ms, subsampling_factor);
  }

  const Model *GetModel() const { return model_.get(); }

 private:
  RecognizerConfig config_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<Decoder> decoder_;
  Endpoint endpoint_;
  SymbolTable sym_;
};

Recognizer::Recognizer(const RecognizerConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
Recognizer::Recognizer(AAssetManager *mgr, const RecognizerConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

Recognizer::~Recognizer() = default;

std::unique_ptr<Stream> Recognizer::CreateStream() const {
  return impl_->CreateStream();
}

//std::unique_ptr<Stream> Recognizer::CreateStream(
//    const std::vector<std::vector<int32_t>> &context_list) const {
//  return impl_->CreateStream(context_list);
//}
std::unique_ptr<Stream> Recognizer::CreateStream(
    const char* context_list) const {
  return impl_->CreateStream(context_list);
}

bool Recognizer::IsReady(Stream *s) const { return impl_->IsReady(s); }

void Recognizer::DecodeStream(Stream *s) const { impl_->DecodeStream(s); }

bool Recognizer::IsEndpoint(Stream *s) const { return impl_->IsEndpoint(s); }

void Recognizer::Reset(Stream *s) const { impl_->Reset(s); }

RecognitionResult Recognizer::GetResult(Stream *s) const {
  return impl_->GetResult(s);
}

const Model *Recognizer::GetModel() const { return impl_->GetModel(); }

}  // namespace sherpa_ncnn
