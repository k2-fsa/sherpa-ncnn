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

// TODO(fangjun): Add documentation to functions/methods in this file
// and also show how to use them with kotlin, possibly with java.

// If you use ndk, you can find "jni.h" inside
// android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
#include "jni.h"  // NOLINT

#include <strstream>

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#include "sherpa-ncnn/csrc/decode.h"
#include "sherpa-ncnn/csrc/features.h"
#include "sherpa-ncnn/csrc/model.h"
#include "sherpa-ncnn/csrc/symbol-table.h"
#include "sherpa-ncnn/csrc/wave-reader.h"

#define SHERPA_EXTERN_C extern "C"

namespace sherpa_ncnn {

class SherpaNcnn {
 public:
  SherpaNcnn(AAssetManager *mgr, const ModelConfig &model_config,
             const knf::FbankOptions &fbank_config)
      : model_(Model::Create(mgr, model_config)),
        feature_extractor_(std::make_unique<FeatureExtractor>(fbank_config)),
        sym_(mgr, model_config.tokens) {
    Reset();
  }

  void DecodeSamples(float sample_rate, const float *samples, int32_t n) {
    feature_extractor_->AcceptWaveform(sample_rate, samples, n);
    Decode();
  }

  void InputFinished() {
    feature_extractor_->InputFinished();
    Decode();
  }

  std::string GetText() const {
    int32_t context_size = model_->ContextSize();

    std::string text;
    for (int32_t i = context_size; i != static_cast<int32_t>(hyp_.size());
         ++i) {
      text += sym_[hyp_[i]];
    }
    return text;
  }

  void Reset() {
    feature_extractor_->Reset();
    num_processed_ = 0;
    states_.clear();

    int32_t context_size = model_->ContextSize();
    int32_t blank_id = 0;

    ncnn::Mat decoder_input(context_size);
    for (int32_t i = 0; i != context_size; ++i) {
      static_cast<int32_t *>(decoder_input)[i] = blank_id;
    }

    decoder_out_ = model_->RunDecoder(decoder_input);

    hyp_.resize(context_size, 0);
  }

 private:
  void Decode() {
    int32_t segment = model_->Segment();
    int32_t offset = model_->Offset();

    ncnn::Mat encoder_out;
    while (feature_extractor_->NumFramesReady() - num_processed_ >= segment) {
      ncnn::Mat features =
          feature_extractor_->GetFrames(num_processed_, segment);
      num_processed_ += offset;

      std::tie(encoder_out, states_) = model_->RunEncoder(features, states_);

      GreedySearch(model_.get(), encoder_out, &decoder_out_, &hyp_);
    }
  }

 private:
  std::unique_ptr<Model> model_;
  std::unique_ptr<FeatureExtractor> feature_extractor_;
  sherpa_ncnn::SymbolTable sym_;

  std::vector<int32_t> hyp_;
  ncnn::Mat decoder_out_;
  std::vector<ncnn::Mat> states_;

  // number of processed frames
  int32_t num_processed_ = 0;
};

static ModelConfig GetModelConfig(JNIEnv *env, jobject config) {
  ModelConfig model_config;

  jclass cls = env->GetObjectClass(config);

  jfieldID fid = env->GetFieldID(cls, "encoderParam", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  model_config.encoder_param = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "encoderBin", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  model_config.encoder_bin = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "decoderParam", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  model_config.decoder_param = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "decoderBin", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  model_config.decoder_bin = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "joinerParam", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  model_config.joiner_param = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "joinerBin", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  model_config.joiner_bin = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "tokens", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  model_config.tokens = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "numThreads", "I");

  int32_t num_threads = env->GetIntField(config, fid);
  model_config.encoder_opt.num_threads = num_threads;
  model_config.decoder_opt.num_threads = num_threads;
  model_config.joiner_opt.num_threads = num_threads;

  fid = env->GetFieldID(cls, "useGPU", "Z");
  model_config.use_vulkan_compute = env->GetBooleanField(config, fid);

  return model_config;
}

static knf::FbankOptions GetFbankOptions(JNIEnv *env, jobject opts) {
  jclass cls = env->GetObjectClass(opts);
  jfieldID fid;

  // https://docs.oracle.com/javase/7/docs/technotes/guides/jni/spec/types.html
  // https://courses.cs.washington.edu/courses/cse341/99wi/java/tutorial/native1.1/implementing/field.html

  knf::FbankOptions fbank_opts;

  fid = env->GetFieldID(cls, "useEnergy", "Z");
  fbank_opts.use_energy = env->GetBooleanField(opts, fid);

  fid = env->GetFieldID(cls, "energyFloor", "F");
  fbank_opts.energy_floor = env->GetFloatField(opts, fid);

  fid = env->GetFieldID(cls, "rawEnergy", "Z");
  fbank_opts.raw_energy = env->GetBooleanField(opts, fid);

  fid = env->GetFieldID(cls, "htkCompat", "Z");
  fbank_opts.htk_compat = env->GetBooleanField(opts, fid);

  fid = env->GetFieldID(cls, "useLogFbank", "Z");
  fbank_opts.use_log_fbank = env->GetBooleanField(opts, fid);

  fid = env->GetFieldID(cls, "usePower", "Z");
  fbank_opts.use_power = env->GetBooleanField(opts, fid);

  fid = env->GetFieldID(cls, "frameOpts",
                        "Lcom/k2fsa/sherpa/ncnn/FrameExtractionOptions;");

  jobject frame_opts = env->GetObjectField(opts, fid);
  jclass frame_opts_cls = env->GetObjectClass(frame_opts);

  fid = env->GetFieldID(frame_opts_cls, "sampFreq", "F");
  fbank_opts.frame_opts.samp_freq = env->GetFloatField(frame_opts, fid);

  fid = env->GetFieldID(frame_opts_cls, "frameShiftMs", "F");
  fbank_opts.frame_opts.frame_shift_ms = env->GetFloatField(frame_opts, fid);

  fid = env->GetFieldID(frame_opts_cls, "frameLengthMs", "F");
  fbank_opts.frame_opts.frame_length_ms = env->GetFloatField(frame_opts, fid);

  fid = env->GetFieldID(frame_opts_cls, "dither", "F");
  fbank_opts.frame_opts.dither = env->GetFloatField(frame_opts, fid);

  fid = env->GetFieldID(frame_opts_cls, "preemphCoeff", "F");
  fbank_opts.frame_opts.preemph_coeff = env->GetFloatField(frame_opts, fid);

  fid = env->GetFieldID(frame_opts_cls, "removeDcOffset", "Z");
  fbank_opts.frame_opts.remove_dc_offset =
      env->GetBooleanField(frame_opts, fid);

  fid = env->GetFieldID(frame_opts_cls, "windowType", "Ljava/lang/String;");
  jstring window_type = (jstring)env->GetObjectField(frame_opts, fid);
  const char *p_window_type = env->GetStringUTFChars(window_type, nullptr);
  fbank_opts.frame_opts.window_type = p_window_type;
  env->ReleaseStringUTFChars(window_type, p_window_type);

  fid = env->GetFieldID(frame_opts_cls, "roundToPowerOfTwo", "Z");
  fbank_opts.frame_opts.round_to_power_of_two =
      env->GetBooleanField(frame_opts, fid);

  fid = env->GetFieldID(frame_opts_cls, "blackmanCoeff", "F");
  fbank_opts.frame_opts.blackman_coeff = env->GetFloatField(frame_opts, fid);

  fid = env->GetFieldID(frame_opts_cls, "snipEdges", "Z");
  fbank_opts.frame_opts.snip_edges = env->GetBooleanField(frame_opts, fid);

  fid = env->GetFieldID(frame_opts_cls, "maxFeatureVectors", "I");
  fbank_opts.frame_opts.max_feature_vectors = env->GetIntField(frame_opts, fid);

  fid = env->GetFieldID(cls, "melOpts",
                        "Lcom/k2fsa/sherpa/ncnn/MelBanksOptions;");
  jobject mel_opts = env->GetObjectField(opts, fid);
  jclass mel_opts_cls = env->GetObjectClass(mel_opts);

  fid = env->GetFieldID(mel_opts_cls, "numBins", "I");
  fbank_opts.mel_opts.num_bins = env->GetIntField(mel_opts, fid);

  fid = env->GetFieldID(mel_opts_cls, "lowFreq", "F");
  fbank_opts.mel_opts.low_freq = env->GetFloatField(mel_opts, fid);

  fid = env->GetFieldID(mel_opts_cls, "highFreq", "F");
  fbank_opts.mel_opts.high_freq = env->GetFloatField(mel_opts, fid);

  fid = env->GetFieldID(mel_opts_cls, "vtlnLow", "F");
  fbank_opts.mel_opts.vtln_low = env->GetFloatField(mel_opts, fid);

  fid = env->GetFieldID(mel_opts_cls, "vtlnHigh", "F");
  fbank_opts.mel_opts.vtln_high = env->GetFloatField(mel_opts, fid);

  fid = env->GetFieldID(mel_opts_cls, "debugMel", "Z");
  fbank_opts.mel_opts.debug_mel = env->GetBooleanField(mel_opts, fid);

  fid = env->GetFieldID(mel_opts_cls, "htkMode", "Z");
  fbank_opts.mel_opts.htk_mode = env->GetBooleanField(mel_opts, fid);

  return fbank_opts;
}

}  // namespace sherpa_ncnn

SHERPA_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_new(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _model_config,
    jobject _fbank_config) {
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    NCNN_LOGE("Failed to get asset manager: %p", mgr);
  }

  sherpa_ncnn::ModelConfig model_config =
      sherpa_ncnn::GetModelConfig(env, _model_config);

  knf::FbankOptions fbank_opts =
      sherpa_ncnn::GetFbankOptions(env, _fbank_config);

  auto model = new sherpa_ncnn::SherpaNcnn(mgr, model_config, fbank_opts);

  return (jlong)model;
}

SHERPA_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_delete(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_ncnn::SherpaNcnn *>(ptr);
}

SHERPA_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_reset(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  reinterpret_cast<sherpa_ncnn::SherpaNcnn *>(ptr)->Reset();
}

SHERPA_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_decodeSamples(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jfloat sample_rate) {
  auto model = reinterpret_cast<sherpa_ncnn::SherpaNcnn *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  model->DecodeSamples(sample_rate, p, n);

  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}

SHERPA_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_inputFinished(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  reinterpret_cast<sherpa_ncnn::SherpaNcnn *>(ptr)->InputFinished();
}

SHERPA_EXTERN_C
JNIEXPORT jstring JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_getText(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  // see
  // https://stackoverflow.com/questions/11621449/send-c-string-to-java-via-jni
  auto text = reinterpret_cast<sherpa_ncnn::SherpaNcnn *>(ptr)->GetText();
  return env->NewStringUTF(text.c_str());
}

SHERPA_EXTERN_C
JNIEXPORT jfloatArray JNICALL
Java_com_k2fsa_sherpa_ncnn_WaveReader_00024Companion_readWave(
    JNIEnv *env, jclass /*cls*/, jobject asset_manager, jstring filename,
    jfloat expected_sample_rate) {
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    NCNN_LOGE("Failed to get asset manager: %p", mgr);
    return nullptr;
  }
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);

  AAsset *asset = AAssetManager_open(mgr, p_filename, AASSET_MODE_BUFFER);
  size_t asset_length = AAsset_getLength(asset);
  std::vector<char> buffer(asset_length);
  AAsset_read(asset, buffer.data(), asset_length);

  std::istrstream is(buffer.data(), asset_length);

  bool is_ok = false;
  std::vector<float> samples =
      sherpa_ncnn::ReadWave(is, expected_sample_rate, &is_ok);

  AAsset_close(asset);
  env->ReleaseStringUTFChars(filename, p_filename);

  if (!is_ok) {
    return nullptr;
  }

  jfloatArray ans = env->NewFloatArray(samples.size());
  env->SetFloatArrayRegion(ans, 0, samples.size(), samples.data());
  return ans;
}
