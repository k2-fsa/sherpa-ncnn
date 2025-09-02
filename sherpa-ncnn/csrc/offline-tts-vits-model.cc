// sherpa-ncnn/csrc/offline-tts-vits-model.cc
//
// Copyright 2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-tts-vits-model.h"

#include <math.h>

#include "net.h"

namespace sherpa_ncnn {

// this function is is modified from nihui's implementation
static ncnn::Mat PathAttentionImpl(const ncnn::Mat &logw, const ncnn::Mat &m_p,
                                   ncnn::Mat &logs_p, float noise_scale,
                                   float speed) {
  float length_scale = 1 / speed;

  const int x_lengths = logw.w;

  const int depth = m_p.h;

  std::vector<int> w_ceil(x_lengths);
  int y_lengths = 0;
  for (int i = 0; i < x_lengths; i++) {
    w_ceil[i] = (int)ceilf(expf(logw[i]) * length_scale);
    y_lengths += w_ceil[i];
  }

  ncnn::Mat z_p;

  z_p.create(y_lengths, depth);

  for (int i = 0; i < depth; i++) {
    const float *m_p_ptr = m_p.row(i);
    const float *logs_p_ptr = logs_p.row(i);
    float *ptr = z_p.row(i);

    for (int j = 0; j < x_lengths; j++) {
      const float m = m_p_ptr[j];
      const float nl = expf(logs_p_ptr[j]) * noise_scale;
      const int duration = w_ceil[j];

      for (int k = 0; k < duration; k++) {
        ptr[k] = m + (rand() / (float)RAND_MAX) * nl;
      }
      ptr += duration;
    }
  }

  return z_p;
}

// this class is written by nihui
class relative_embeddings_k_module : public ncnn::Layer {
 public:
  relative_embeddings_k_module() { one_blob_only = true; }

  virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob,
                      const ncnn::Option &opt) const {
    const int window_size = 4;

    const int wsize = bottom_blob.w;
    const int len = bottom_blob.h;
    const int num_heads = bottom_blob.c;

    top_blob.create(len, len, num_heads);

    top_blob.fill(0.f);

#pragma omp parallel for
    for (int q = 0; q < num_heads; q++) {
      const ncnn::Mat x0 = bottom_blob.channel(q);
      ncnn::Mat out0 = top_blob.channel(q);

      for (int i = 0; i < len; i++) {
        const float *xptr = x0.row(i) + std::max(0, window_size - i);
        float *outptr = out0.row(i) + std::max(i - window_size, 0);
        const int wsize2 = std::min(len, i - window_size + wsize) -
                           std::max(i - window_size, 0);
        for (int j = 0; j < wsize2; j++) {
          *outptr++ = *xptr++;
        }
      }
    }

    return 0;
  }
};

DEFINE_LAYER_CREATOR(relative_embeddings_k_module)

// this class is written by nihui
class relative_embeddings_v_module : public ncnn::Layer {
 public:
  relative_embeddings_v_module() { one_blob_only = true; }

  virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob,
                      const ncnn::Option &opt) const {
    const int window_size = 4;

    const int wsize = window_size * 2 + 1;
    const int len = bottom_blob.h;
    const int num_heads = bottom_blob.c;

    top_blob.create(wsize, len, num_heads);

    top_blob.fill(0.f);

#pragma omp parallel for
    for (int q = 0; q < num_heads; q++) {
      const ncnn::Mat x0 = bottom_blob.channel(q);
      ncnn::Mat out0 = top_blob.channel(q);

      for (int i = 0; i < len; i++) {
        const float *xptr = x0.row(i) + std::max(i - window_size, 0);
        float *outptr = out0.row(i) + std::max(0, window_size - i);
        const int wsize2 = std::min(len, i - window_size + wsize) -
                           std::max(i - window_size, 0);
        for (int j = 0; j < wsize2; j++) {
          *outptr++ = *xptr++;
        }
      }
    }

    return 0;
  }
};

DEFINE_LAYER_CREATOR(relative_embeddings_v_module)

// this class is from by nihui
class piecewise_rational_quadratic_transform_module : public ncnn::Layer {
 public:
  piecewise_rational_quadratic_transform_module() { one_blob_only = false; }

  virtual int forward(const std::vector<ncnn::Mat> &bottom_blobs,
                      std::vector<ncnn::Mat> &top_blobs,
                      const ncnn::Option &opt) const {
    const ncnn::Mat &h = bottom_blobs[0];
    const ncnn::Mat &x1 = bottom_blobs[1];
    ncnn::Mat &outputs = top_blobs[0];

    const int num_bins = 10;
    const int filter_channels = 192;
    const bool reverse = true;
    const float tail_bound = 5.0f;
    const float DEFAULT_MIN_BIN_WIDTH = 1e-3f;
    const float DEFAULT_MIN_BIN_HEIGHT = 1e-3f;
    const float DEFAULT_MIN_DERIVATIVE = 1e-3f;

    // x1 shape: (w=N, h=1, c=1), h shape (w=29*N, h=1, c=1) due to Fortran
    // layout
    const int batch_size = x1.w;
    const int h_params_per_item = 2 * num_bins + (num_bins - 1);  // 29

    outputs = x1.clone();

    float *out_ptr = outputs;

    for (int i = 0; i < batch_size; ++i) {
      const float current_x = ((const float *)x1)[i];

      const float *h_data = h.row(i);

      if (current_x < -tail_bound || current_x > tail_bound) {
        continue;
      }

      std::vector<float> unnormalized_widths(num_bins);
      std::vector<float> unnormalized_heights(num_bins);
      std::vector<float> unnormalized_derivatives(num_bins + 1);

      const float inv_sqrt_filter_channels = 1.0f / sqrtf(filter_channels);
      for (int j = 0; j < num_bins; ++j) {
        unnormalized_widths[j] = h_data[j] * inv_sqrt_filter_channels;
      }
      for (int j = 0; j < num_bins; ++j) {
        unnormalized_heights[j] =
            h_data[num_bins + j] * inv_sqrt_filter_channels;
      }
      for (int j = 0; j < num_bins - 1; ++j) {
        unnormalized_derivatives[j + 1] = h_data[2 * num_bins + j];
      }

      const float constant = logf(expf(1.f - DEFAULT_MIN_DERIVATIVE) - 1.f);
      unnormalized_derivatives[0] = constant;
      unnormalized_derivatives[num_bins] = constant;

      const float left = -tail_bound, right = tail_bound;
      const float bottom = -tail_bound, top = tail_bound;

      std::vector<float> widths(num_bins);
      float w_max = -INFINITY;
      for (float val : unnormalized_widths) w_max = std::max(w_max, val);
      float w_sum = 0.f;
      for (int j = 0; j < num_bins; ++j) {
        widths[j] = expf(unnormalized_widths[j] - w_max);
        w_sum += widths[j];
      }
      for (int j = 0; j < num_bins; ++j) {
        widths[j] =
            DEFAULT_MIN_BIN_WIDTH +
            (1.f - DEFAULT_MIN_BIN_WIDTH * num_bins) * (widths[j] / w_sum);
      }

      std::vector<float> cumwidths(num_bins + 1);
      cumwidths[0] = left;
      float current_w_sum = 0.f;
      for (int j = 0; j < num_bins - 1; ++j) {
        current_w_sum += widths[j];
        cumwidths[j + 1] = left + (right - left) * current_w_sum;
      }
      cumwidths[num_bins] = right;

      std::vector<float> heights(num_bins);
      float h_max = -INFINITY;
      for (float val : unnormalized_heights) h_max = std::max(h_max, val);
      float h_sum = 0.f;
      for (int j = 0; j < num_bins; ++j) {
        heights[j] = expf(unnormalized_heights[j] - h_max);
        h_sum += heights[j];
      }
      for (int j = 0; j < num_bins; ++j) {
        heights[j] =
            DEFAULT_MIN_BIN_HEIGHT +
            (1.f - DEFAULT_MIN_BIN_HEIGHT * num_bins) * (heights[j] / h_sum);
      }

      std::vector<float> cumheights(num_bins + 1);
      cumheights[0] = bottom;
      float current_h_sum = 0.f;
      for (int j = 0; j < num_bins - 1; ++j) {
        current_h_sum += heights[j];
        cumheights[j + 1] = bottom + (top - bottom) * current_h_sum;
      }
      cumheights[num_bins] = top;

      std::vector<float> derivatives(num_bins + 1);
      for (int j = 0; j < num_bins + 1; ++j) {
        float x = unnormalized_derivatives[j];
        derivatives[j] =
            DEFAULT_MIN_DERIVATIVE +
            (x > 0 ? x + logf(1.f + expf(-x)) : logf(1.f + expf(x)));
      }

      int bin_idx = 0;
      if (reverse) {
        auto it =
            std::upper_bound(cumheights.begin(), cumheights.end(), current_x);
        bin_idx = std::distance(cumheights.begin(), it) - 1;
      } else {
        auto it =
            std::upper_bound(cumwidths.begin(), cumwidths.end(), current_x);
        bin_idx = std::distance(cumwidths.begin(), it) - 1;
      }
      bin_idx = std::max(0, std::min(bin_idx, num_bins - 1));

      const float input_cumwidths = cumwidths[bin_idx];
      const float input_bin_widths =
          cumwidths[bin_idx + 1] - cumwidths[bin_idx];
      const float input_cumheights = cumheights[bin_idx];
      const float input_heights = cumheights[bin_idx + 1] - cumheights[bin_idx];
      const float input_derivatives = derivatives[bin_idx];
      const float input_derivatives_plus_one = derivatives[bin_idx + 1];
      const float delta = input_heights / input_bin_widths;

      if (reverse) {
        float a =
            (current_x - input_cumheights) *
                (input_derivatives + input_derivatives_plus_one - 2 * delta) +
            input_heights * (delta - input_derivatives);
        float b =
            input_heights * input_derivatives -
            (current_x - input_cumheights) *
                (input_derivatives + input_derivatives_plus_one - 2 * delta);
        float c = -delta * (current_x - input_cumheights);
        float discriminant = b * b - 4 * a * c;
        discriminant = std::max(0.f, discriminant);
        float root = (2 * c) / (-b - sqrtf(discriminant));
        out_ptr[i] = root * input_bin_widths + input_cumwidths;
      } else {
        float theta = (current_x - input_cumwidths) / input_bin_widths;
        float theta_one_minus_theta = theta * (1 - theta);
        float numerator =
            input_heights *
            (delta * theta * theta + input_derivatives * theta_one_minus_theta);
        float denominator =
            delta +
            ((input_derivatives + input_derivatives_plus_one - 2 * delta) *
             theta_one_minus_theta);
        out_ptr[i] = input_cumheights + numerator / denominator;
      }
    }

    return 0;
  }
};

DEFINE_LAYER_CREATOR(piecewise_rational_quadratic_transform_module)

class OfflineTtsVitsModel::Impl {
 public:
  Impl(const OfflineTtsModelConfig &config) : config_(config) { InitNet(); }

  const OfflineTtsVitsModelMetaData &GetMetaData() const { return meta_; }

  std::vector<ncnn::Mat> RunEncoder(const ncnn::Mat &sequence) const {
    ncnn::Extractor ex = enc_p_.create_extractor();

    ex.input("in0", sequence);

    ncnn::Mat x;
    ncnn::Mat m_p;
    ncnn::Mat logs_p;

    ex.extract("out0", x);
    ex.extract("out1", m_p);
    ex.extract("out2", logs_p);

    return {x, m_p, logs_p};
  }

  ncnn::Mat RunDurationPredictor(const ncnn::Mat &x,
                                 const ncnn::Mat &noise) const {
    ncnn::Extractor ex = dp_.create_extractor();

    ex.input("in0", x);
    ex.input("in1", noise);

    ncnn::Mat logw;
    ex.extract("out0", logw);

    return logw;
  }

  ncnn::Mat RunFlow(const ncnn::Mat &z_p) const {
    ncnn::Extractor ex = flow_.create_extractor();

    ex.input("in0", z_p);

    ncnn::Mat z;
    ex.extract("out0", z);

    return z;
  }

  ncnn::Mat RunDecoder(const ncnn::Mat &z) const {
    ncnn::Extractor ex = decoder_.create_extractor();

    ex.input("in0", z);

    ncnn::Mat o;
    ex.extract("out0", o);

    return o;
  }

 private:
  void InitNet() {
    InitEncoderNet();
    InitDurationPredictorNet();
    InitFlowNet();
    InitDecoderNet();
  }

  void InitEncoderNet() {
    enc_p_.opt.num_threads = config_.num_threads;

    // TODO(fangjun): change the module name
    enc_p_.register_custom_layer("en_enc_p_pnnx.relative_embeddings_k_module",
                                 relative_embeddings_k_module_layer_creator);
    enc_p_.register_custom_layer("en_enc_p_pnnx.relative_embeddings_v_module",
                                 relative_embeddings_v_module_layer_creator);

    std::string param = config_.vits.model_dir + "/encoder.ncnn.param";
    std::string bin = config_.vits.model_dir + "/encoder.ncnn.bin";
    enc_p_.load_param(param.c_str());
    enc_p_.load_model(bin.c_str());
  }

  void InitDurationPredictorNet() {
    dp_.opt.num_threads = config_.num_threads;

    dp_.register_custom_layer(
        "piper.train.vits.modules.piecewise_rational_quadratic_transform_"
        "module",
        piecewise_rational_quadratic_transform_module_layer_creator);

    std::string param = config_.vits.model_dir + "/dp.ncnn.param";
    std::string bin = config_.vits.model_dir + "/dp.ncnn.bin";

    dp_.load_param(param.c_str());
    dp_.load_model(bin.c_str());
  }

  void InitFlowNet() {
    flow_.opt.num_threads = config_.num_threads;

    std::string param = config_.vits.model_dir + "/flow.ncnn.param";
    std::string bin = config_.vits.model_dir + "/flow.ncnn.bin";

    flow_.load_param(param.c_str());
    flow_.load_model(bin.c_str());
  }

  void InitDecoderNet() {
    decoder_.opt.num_threads = config_.num_threads;

    std::string param = config_.vits.model_dir + "/decoder.ncnn.param";
    std::string bin = config_.vits.model_dir + "/decoder.ncnn.bin";

    decoder_.load_param(param.c_str());
    decoder_.load_model(bin.c_str());
  }

 private:
  OfflineTtsModelConfig config_;
  OfflineTtsVitsModelMetaData meta_;

  ncnn::Net enc_p_;
  ncnn::Net dp_;
  ncnn::Net flow_;
  ncnn::Net decoder_;
};

OfflineTtsVitsModel::~OfflineTtsVitsModel() = default;

OfflineTtsVitsModel::OfflineTtsVitsModel(const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

const OfflineTtsVitsModelMetaData &OfflineTtsVitsModel::GetMetaData() const {
  return impl_->GetMetaData();
}

std::vector<ncnn::Mat> OfflineTtsVitsModel::RunEncoder(
    const ncnn::Mat &sequence) const {
  return impl_->RunEncoder(sequence);
}

ncnn::Mat OfflineTtsVitsModel::RunDurationPredictor(
    const ncnn::Mat &x, const ncnn::Mat &noise) const {
  return impl_->RunDurationPredictor(x, noise);
}

ncnn::Mat OfflineTtsVitsModel::PathAttention(const ncnn::Mat &logw,
                                             const ncnn::Mat &m_p,
                                             ncnn::Mat &logs_p,
                                             float noise_scale, float speed) {
  return PathAttentionImpl(logw, m_p, logs_p, noise_scale, speed);
}

ncnn::Mat OfflineTtsVitsModel::RunFlow(const ncnn::Mat &z_p) const {
  return impl_->RunFlow(z_p);
}

ncnn::Mat OfflineTtsVitsModel::RunDecoder(const ncnn::Mat &z) const {
  return impl_->RunDecoder(z);
}

}  // namespace sherpa_ncnn
