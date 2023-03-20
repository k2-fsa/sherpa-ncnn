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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-ncnn/csrc/decoder.h"
#include "sherpa-ncnn/csrc/greedy-search-decoder.h"
#include "sherpa-ncnn/csrc/modified-beam-search-decoder.h"

#include <iostream>

namespace sherpa_ncnn {

static RecognitionResult Convert(const DecoderResult &src,
                                 const SymbolTable &sym_table,
								 int32_t frame_shift_ms,
								 int32_t subsampling_factor) {
  RecognitionResult ans;
  ans.tokens.reserve(src.tokens.size());
  ans.words.reserve(src.tokens.size());
  ans.timestamps.reserve(src.timestamps.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];
    text.append(sym);
	ans.words.push_back(sym);
    ans.tokens.push_back(i);
  }

  ans.text = std::move(text);
  ans.tokens = src.tokens;
  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  //std::cout<<"frame_shift="<<frame_shift_s;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
	//std::cout<<",["<<t<<"|"<<time<<"]";
    ans.timestamps.push_back(time);
  }
  //std::cout<<std::endl;
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
  os << "endpoint_config=" << endpoint_config.ToString() << ", ";
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ")";

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

  bool IsReady(Stream *s) const {
    return s->GetNumProcessedFrames() + model_->Segment() < s->NumFramesReady();
  }

  void DecodeStream(Stream *s) const {
    int32_t segment = model_->Segment();
    int32_t offset = model_->Offset();

    ncnn::Mat features = s->GetFrames(s->GetNumProcessedFrames(), segment);
    s->GetNumProcessedFrames() += offset;
    std::vector<ncnn::Mat> states = s->GetStates();

    ncnn::Mat encoder_out;
    std::tie(encoder_out, states) = model_->RunEncoder(features, states);
    s->SetStates(states);

    decoder_->Decode(encoder_out, &s->GetResult());
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

    //return Convert(decoder_result, sym_);
	// Those 2 parameter figured out from sherpa source code
	int32_t frame_shift_ms = 10;
	int32_t subsampling_factor = 4;
    return Convert(decoder_result, sym_, frame_shift_ms, subsampling_factor);
			//config_.feat_config.fbank_opts.frame_opts.frame_shift_ms,
			//model_->SubsamplingFactor());
  }

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

bool Recognizer::IsReady(Stream *s) const { return impl_->IsReady(s); }

void Recognizer::DecodeStream(Stream *s) const { impl_->DecodeStream(s); }

bool Recognizer::IsEndpoint(Stream *s) const { return impl_->IsEndpoint(s); }

void Recognizer::Reset(Stream *s) const { impl_->Reset(s); }

RecognitionResult Recognizer::GetResult(Stream *s) const {
  return impl_->GetResult(s);
}

}  // namespace sherpa_ncnn
