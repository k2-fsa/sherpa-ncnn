// sherpa-ncnn/csrc/offline-tts-vits-model.cc
//
// Copyright 2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-tts-vits-model.h"

#include "net.h"

namespace sherpa_ncnn {

static ncnn::Mat PathAttentionImpl(const ncnn::Mat &logw, const ncnn::Mat &m_p,
                                   ncnn::Mat &logs_p, float noise_scale,
                                   float speed) {
  return {};
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
    return {};
  }

  ncnn::Mat RunFlow(const ncnn::Mat &z_p) const { return {}; }

  ncnn::Mat RunDecoder(const ncnn::Mat &z) const { return {}; }

 private:
  void InitNet() { InitEncoderNet(); }

  void InitEncoderNet() {
    enc_p_.register_custom_layer("en_enc_p_pnnx.relative_embeddings_k_module",
                                 relative_embeddings_k_module_layer_creator);
    enc_p_.register_custom_layer("en_enc_p_pnnx.relative_embeddings_v_module",
                                 relative_embeddings_v_module_layer_creator);

    std::string param = config_.vits.model_dir + "/encoder.ncnn.param";
    std::string bin = config_.vits.model_dir + "/encoder.ncnn.bin";
    enc_p_.load_param(param.c_str());
    enc_p_.load_model(bin.c_str());

    enc_p_.opt.num_threads = config_.num_threads;
  }

 private:
  OfflineTtsModelConfig config_;
  OfflineTtsVitsModelMetaData meta_;

  ncnn::Net enc_p_;
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
