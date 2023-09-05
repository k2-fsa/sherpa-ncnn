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

// TODO(fangjun): Add documentation to functions/methods in this file
// and also show how to use them with kotlin, possibly with java.

// If you use ndk, you can find "jni.h" inside
// android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
#include "jni.h"  // NOLINT

#include <strstream>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#else
#include <fstream>
#endif

#include "sherpa-ncnn/csrc/recognizer.h"
#include "sherpa-ncnn/csrc/wave-reader.h"

#define SHERPA_EXTERN_C extern "C"

namespace sherpa_ncnn {

class SherpaNcnn {
 public:
#if __ANDROID_API__ >= 9
  SherpaNcnn(AAssetManager *mgr, const sherpa_ncnn::RecognizerConfig &config)
      : recognizer_(mgr, config),
        stream_(recognizer_.CreateStream()),
        tail_padding_(16000 * 0.32, 0) {}
#endif

  explicit SherpaNcnn(const sherpa_ncnn::RecognizerConfig &config)
      : recognizer_(config),
        stream_(recognizer_.CreateStream()),
        tail_padding_(16000 * 0.32, 0) {}

  void AcceptWaveform(float sample_rate, const float *samples, int32_t n) {
    stream_->AcceptWaveform(sample_rate, samples, n);
  }

  void InputFinished() {
    stream_->AcceptWaveform(16000, tail_padding_.data(), tail_padding_.size());
    stream_->InputFinished();
  }

  bool IsReady() const { return recognizer_.IsReady(stream_.get()); }

  void DecodeStream() const { return recognizer_.DecodeStream(stream_.get()); }

  const std::string GetText() const {
    auto result = recognizer_.GetResult(stream_.get());
    return result.text;
  }

  bool IsEndpoint() const { return recognizer_.IsEndpoint(stream_.get()); }

  void Reset(bool recreate) {
    if (recreate) {
      stream_ = recognizer_.CreateStream();
    } else {
      recognizer_.Reset(stream_.get());
    }
  }

 private:
  Recognizer recognizer_;
  std::unique_ptr<Stream> stream_;
  std::vector<float> tail_padding_;
};

static FeatureExtractorConfig GetFeatureExtractorConfig(JNIEnv *env,
                                                        jobject config) {
  FeatureExtractorConfig ans;

  jclass cls = env->GetObjectClass(config);

  jfieldID fid = env->GetFieldID(
      cls, "featConfig", "Lcom/k2fsa/sherpa/ncnn/FeatureExtractorConfig;");
  jobject feat_config = env->GetObjectField(config, fid);
  jclass feat_config_cls = env->GetObjectClass(feat_config);

  fid = env->GetFieldID(feat_config_cls, "sampleRate", "F");
  ans.sampling_rate = env->GetFloatField(feat_config, fid);

  fid = env->GetFieldID(feat_config_cls, "featureDim", "I");
  ans.feature_dim = env->GetIntField(feat_config, fid);

  return ans;
}

static ModelConfig GetModelConfig(JNIEnv *env, jobject config) {
  ModelConfig ans;

  jclass cls = env->GetObjectClass(config);

  jfieldID fid = env->GetFieldID(cls, "modelConfig",
                                 "Lcom/k2fsa/sherpa/ncnn/ModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);

  fid = env->GetFieldID(model_config_cls, "encoderParam", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(model_config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.encoder_param = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "encoderBin", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.encoder_bin = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "decoderParam", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.decoder_param = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "decoderBin", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.decoder_bin = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "joinerParam", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.joiner_param = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "joinerBin", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.joiner_bin = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "tokens", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.tokens = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "numThreads", "I");

  int32_t num_threads = env->GetIntField(model_config, fid);
  ans.encoder_opt.num_threads = num_threads;
  ans.decoder_opt.num_threads = num_threads;
  ans.joiner_opt.num_threads = num_threads;

  fid = env->GetFieldID(model_config_cls, "useGPU", "Z");
  ans.use_vulkan_compute = env->GetBooleanField(model_config, fid);

  return ans;
}

static DecoderConfig GetDecoderConfig(JNIEnv *env, jobject config) {
  DecoderConfig ans;

  jclass cls = env->GetObjectClass(config);

  jfieldID fid = env->GetFieldID(cls, "decoderConfig",
                                 "Lcom/k2fsa/sherpa/ncnn/DecoderConfig;");
  jobject decoder_config = env->GetObjectField(config, fid);
  jclass decoder_config_cls = env->GetObjectClass(decoder_config);

  fid = env->GetFieldID(decoder_config_cls, "method", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(decoder_config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.method = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(decoder_config_cls, "numActivePaths", "I");
  ans.num_active_paths = env->GetIntField(decoder_config, fid);

  return ans;
#if 0
  fid = env->GetFieldID(cls, "enableEndpoint", "Z");
  decoder_config.enable_endpoint = env->GetBooleanField(config, fid);

  fid = env->GetFieldID(cls, "endpointConfig",
                        "Lcom/k2fsa/sherpa/ncnn/EndpointConfig;");
  jobject endpoint_config = env->GetObjectField(config, fid);
  jclass endpoint_config_cls = env->GetObjectClass(endpoint_config);

  fid = env->GetFieldID(endpoint_config_cls, "rule1",
                        "Lcom/k2fsa/sherpa/ncnn/EndpointRule;");
  jobject rule1 = env->GetObjectField(endpoint_config, fid);
  jclass rule_class = env->GetObjectClass(rule1);

  fid = env->GetFieldID(endpoint_config_cls, "rule2",
                        "Lcom/k2fsa/sherpa/ncnn/EndpointRule;");
  jobject rule2 = env->GetObjectField(endpoint_config, fid);

  fid = env->GetFieldID(endpoint_config_cls, "rule3",
                        "Lcom/k2fsa/sherpa/ncnn/EndpointRule;");
  jobject rule3 = env->GetObjectField(endpoint_config, fid);

  fid = env->GetFieldID(rule_class, "mustContainNonSilence", "Z");
  decoder_config.endpoint_config.rule1.must_contain_nonsilence =
      env->GetBooleanField(rule1, fid);
  decoder_config.endpoint_config.rule2.must_contain_nonsilence =
      env->GetBooleanField(rule2, fid);
  decoder_config.endpoint_config.rule3.must_contain_nonsilence =
      env->GetBooleanField(rule3, fid);

  fid = env->GetFieldID(rule_class, "minTrailingSilence", "F");
  decoder_config.endpoint_config.rule1.min_trailing_silence =
      env->GetFloatField(rule1, fid);
  decoder_config.endpoint_config.rule2.min_trailing_silence =
      env->GetFloatField(rule2, fid);
  decoder_config.endpoint_config.rule3.min_trailing_silence =
      env->GetFloatField(rule3, fid);

  fid = env->GetFieldID(rule_class, "minUtteranceLength", "F");
  decoder_config.endpoint_config.rule1.min_utterance_length =
      env->GetFloatField(rule1, fid);
  decoder_config.endpoint_config.rule2.min_utterance_length =
      env->GetFloatField(rule2, fid);
  decoder_config.endpoint_config.rule3.min_utterance_length =
      env->GetFloatField(rule3, fid);
#endif
}

static RecognizerConfig ParseConfig(JNIEnv *env, jobject _config) {
  sherpa_ncnn::RecognizerConfig config;
  config.feat_config = sherpa_ncnn::GetFeatureExtractorConfig(env, _config);
  config.model_config = sherpa_ncnn::GetModelConfig(env, _config);
  config.decoder_config = sherpa_ncnn::GetDecoderConfig(env, _config);

  // for endpointing

  jclass cls = env->GetObjectClass(_config);
  jfieldID fid = env->GetFieldID(cls, "enableEndpoint", "Z");
  config.enable_endpoint = env->GetBooleanField(_config, fid);

  fid = env->GetFieldID(cls, "rule1MinTrailingSilence", "F");
  config.endpoint_config.rule1.min_trailing_silence =
      env->GetFloatField(_config, fid);

  fid = env->GetFieldID(cls, "rule2MinTrailingSilence", "F");
  config.endpoint_config.rule2.min_trailing_silence =
      env->GetFloatField(_config, fid);

  fid = env->GetFieldID(cls, "rule3MinUtteranceLength", "F");
  config.endpoint_config.rule3.min_utterance_length =
      env->GetFloatField(_config, fid);

  fid = env->GetFieldID(cls, "hotwordsFile", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(_config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  config.hotwords_file = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "hotwordsScore", "F");
  config.hotwords_score = env->GetFloatField(_config, fid);

  NCNN_LOGE("------config------\n%s\n", config.ToString().c_str());

  return config;
}

}  // namespace sherpa_ncnn

SHERPA_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_newFromFile(
    JNIEnv *env, jobject /*obj*/, jobject _config) {
  sherpa_ncnn::RecognizerConfig config = sherpa_ncnn::ParseConfig(env, _config);
  auto model = new sherpa_ncnn::SherpaNcnn(config);

  return (jlong)model;
}

SHERPA_EXTERN_C
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_newFromAsset(
    JNIEnv *env, jobject /*obj*/, jobject asset_manager, jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    NCNN_LOGE("Failed to get asset manager: %p", mgr);
  }
#endif

  sherpa_ncnn::RecognizerConfig config = sherpa_ncnn::ParseConfig(env, _config);
  auto model = new sherpa_ncnn::SherpaNcnn(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)model;
}

SHERPA_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_delete(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_ncnn::SherpaNcnn *>(ptr);
}

SHERPA_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_decode(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_ncnn::SherpaNcnn *>(ptr);
  model->DecodeStream();
}

SHERPA_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_reset(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jboolean recreate) {
  auto model = reinterpret_cast<sherpa_ncnn::SherpaNcnn *>(ptr);
  model->Reset(recreate);
}

SHERPA_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_isEndpoint(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_ncnn::SherpaNcnn *>(ptr);
  return model->IsEndpoint();
}

SHERPA_EXTERN_C
JNIEXPORT bool JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_isReady(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto model = reinterpret_cast<sherpa_ncnn::SherpaNcnn *>(ptr);
  return model->IsReady();
}

SHERPA_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_acceptWaveform(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jfloat sample_rate) {
  auto model = reinterpret_cast<sherpa_ncnn::SherpaNcnn *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  model->AcceptWaveform(sample_rate, p, n);

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
  const char *p_filename = env->GetStringUTFChars(filename, nullptr);
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    NCNN_LOGE("Failed to get asset manager: %p", mgr);
    return nullptr;
  }

  AAsset *asset = AAssetManager_open(mgr, p_filename, AASSET_MODE_BUFFER);
  size_t asset_length = AAsset_getLength(asset);
  std::vector<char> buffer(asset_length);
  AAsset_read(asset, buffer.data(), asset_length);

  std::istrstream is(buffer.data(), asset_length);
#else
  std::ifstream is(p_filename, std::ios::binary);
#endif

  bool is_ok = false;
  std::vector<float> samples =
      sherpa_ncnn::ReadWave(is, expected_sample_rate, &is_ok);

#if __ANDROID_API__ >= 9
  AAsset_close(asset);
#endif
  env->ReleaseStringUTFChars(filename, p_filename);

  if (!is_ok) {
    return nullptr;
  }

  jfloatArray ans = env->NewFloatArray(samples.size());
  env->SetFloatArrayRegion(ans, 0, samples.size(), samples.data());
  return ans;
}
