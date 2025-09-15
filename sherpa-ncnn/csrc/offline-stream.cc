// sherpa-ncnn/csrc/offline-stream.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-stream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "mat.h"  // NOLINT
#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/offline-recognizer.h"
#include "sherpa-ncnn/csrc/resample.h"

namespace sherpa_ncnn {

class OfflineStream::Impl {
 public:
  explicit Impl(const FeatureExtractorConfig &config) : config_(config) {
    knf::FbankOptions opts;
    opts.frame_opts.dither = config.dither;
    opts.frame_opts.snip_edges = config.snip_edges;
    opts.frame_opts.samp_freq = config.sampling_rate;
    opts.frame_opts.frame_shift_ms = config.frame_shift_ms;
    opts.frame_opts.frame_length_ms = config.frame_length_ms;
    opts.frame_opts.remove_dc_offset = config.remove_dc_offset;
    opts.frame_opts.window_type = config.window_type;

    opts.mel_opts.num_bins = config.feature_dim;

    opts.mel_opts.high_freq = config.high_freq;
    opts.mel_opts.low_freq = config.low_freq;

    opts.mel_opts.is_librosa = config.is_librosa;

    fbank_ = std::make_unique<knf::OnlineFbank>(opts);
  }

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    if (config_.normalize_samples) {
      AcceptWaveformImpl(sampling_rate, waveform, n);
    } else {
      std::vector<float> buf(n);
      for (int32_t i = 0; i != n; ++i) {
        buf[i] = waveform[i] * 32768;
      }
      AcceptWaveformImpl(sampling_rate, buf.data(), n);
    }
  }

  void AcceptWaveformImpl(int32_t sampling_rate, const float *waveform,
                          int32_t n) {
    if (sampling_rate != config_.sampling_rate) {
      SHERPA_NCNN_LOGE(
          "Creating a resampler:\n"
          "   in_sample_rate: %d\n"
          "   output_sample_rate: %d\n",
          sampling_rate, static_cast<int32_t>(config_.sampling_rate));

      float min_freq = std::min<int32_t>(sampling_rate, config_.sampling_rate);
      float lowpass_cutoff = 0.99 * 0.5 * min_freq;

      int32_t lowpass_filter_width = 6;
      auto resampler = std::make_unique<LinearResample>(
          sampling_rate, config_.sampling_rate, lowpass_cutoff,
          lowpass_filter_width);
      std::vector<float> samples;
      resampler->Resample(waveform, n, true, &samples);

      fbank_->AcceptWaveform(config_.sampling_rate, samples.data(),
                             samples.size());
      fbank_->InputFinished();
      return;
    }  // if (sampling_rate != config_.sampling_rate)

    fbank_->AcceptWaveform(sampling_rate, waveform, n);
    fbank_->InputFinished();
  }

  int32_t FeatureDim() const { return config_.feature_dim; }

  ncnn::Mat GetFrames() const {
    int32_t n = fbank_->NumFramesReady();
    assert(n > 0 && "Please first call AcceptWaveform()");

    int32_t feature_dim = FeatureDim();

    ncnn::Mat features;
    features.create(feature_dim, n);

    for (int32_t i = 0; i != n; ++i) {
      const float *f = fbank_->GetFrame(i);
      std::copy(f, f + feature_dim, features.row(i));
    }

    return features;
  }

  void SetResult(const OfflineRecognizerResult &r) { r_ = r; }

  const OfflineRecognizerResult &GetResult() const { return r_; }

 private:
  FeatureExtractorConfig config_;
  std::unique_ptr<knf::OnlineFbank> fbank_;
  OfflineRecognizerResult r_;
};

OfflineStream::OfflineStream(const FeatureExtractorConfig &config /*= {}*/)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineStream::~OfflineStream() = default;

void OfflineStream::AcceptWaveform(int32_t sampling_rate, const float *waveform,
                                   int32_t n) const {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

int32_t OfflineStream::FeatureDim() const { return impl_->FeatureDim(); }

ncnn::Mat OfflineStream::GetFrames() const { return impl_->GetFrames(); }

void OfflineStream::SetResult(const OfflineRecognizerResult &r) {
  impl_->SetResult(r);
}

const OfflineRecognizerResult &OfflineStream::GetResult() const {
  return impl_->GetResult();
}
std::string OfflineRecognizerResult::AsJsonString() const {
  std::ostringstream os;
  os << "{";

  os << "\"lang\""
     << ": ";
  os << std::quoted(lang) << ", ";

  os << "\"emotion\""
     << ": ";
  os << std::quoted(emotion) << ", ";

  os << "\"event\""
     << ": ";
  os << std::quoted(event) << ", ";

  os << "\"text\""
     << ": ";
  os << std::quoted(text) << ", ";

  os << "\""
     << "timestamps"
     << "\""
     << ": ";
  os << "[";

  std::string sep = "";
  for (auto t : timestamps) {
    os << sep << std::fixed << std::setprecision(2) << t;
    sep = ", ";
  }
  os << "], ";

  os << "\""
     << "tokens"
     << "\""
     << ":";
  os << "[";

  sep = "";
  auto oldFlags = os.flags();
  for (const auto &t : tokens) {
    if (t.size() == 1 && static_cast<uint8_t>(t[0]) > 0x7f) {
      const uint8_t *p = reinterpret_cast<const uint8_t *>(t.c_str());
      os << sep << "\""
         << "<0x" << std::hex << std::uppercase << static_cast<uint32_t>(p[0])
         << ">"
         << "\"";
      os.flags(oldFlags);
    } else {
      os << sep << std::quoted(t);
    }
    sep = ", ";
  }
  os << "]";

  os << "}";

  return os.str();
}
}  // namespace sherpa_ncnn
