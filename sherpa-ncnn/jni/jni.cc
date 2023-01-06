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

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#include "sherpa-ncnn/csrc/recognizer.h"
#include "sherpa-ncnn/csrc/wave-reader.h"

#define SHERPA_EXTERN_C extern "C"

namespace sherpa_ncnn {

class SherpaNcnn {
 public:
  SherpaNcnn(AAssetManager *mgr,
             const sherpa_ncnn::DecoderConfig &decoder_config,
             const ModelConfig &model_config,
             const knf::FbankOptions &fbank_opts)
      : recognizer_(mgr, decoder_config, model_config, fbank_opts),
        tail_padding_(16000 * 0.32, 0) {}

  void DecodeSamples(float sample_rate, const float *samples, int32_t n) {
    recognizer_.AcceptWaveform(sample_rate, samples, n);
    recognizer_.Decode();
  }

  void InputFinished() {
    recognizer_.AcceptWaveform(16000, tail_padding_.data(),
                               tail_padding_.size());
    recognizer_.InputFinished();
    recognizer_.Decode();
  }

  const std::string GetText() {
    auto result = recognizer_.GetResult();
    return result.text;
  }

 private:
  sherpa_ncnn::Recognizer recognizer_;
  std::vector<float> tail_padding_;
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

  sherpa_ncnn::DecoderConfig decoder_config;

  knf::FbankOptions fbank_opts =
      sherpa_ncnn::GetFbankOptions(env, _fbank_config);

  auto model = new sherpa_ncnn::SherpaNcnn(mgr, decoder_config, model_config,
                                           fbank_opts);

  return (jlong)model;
}

SHERPA_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_delete(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_ncnn::SherpaNcnn *>(ptr);
}

SHERPA_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_SherpaNcnn_reset(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {}

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
