// sherpa-ncnn/jni/offline-recognizer.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-recognizer.h"

#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/text-utils.h"
#include "sherpa-ncnn/jni/common.h"

namespace sherpa_ncnn {

static OfflineRecognizerConfig GetOfflineConfig(JNIEnv *env, jobject config) {
  OfflineRecognizerConfig ans;

  jclass cls = env->GetObjectClass(config);
  jfieldID fid;

  //---------- decoding ----------
  fid = env->GetFieldID(cls, "decodingMethod", "Ljava/lang/String;");
  jstring s = (jstring)env->GetObjectField(config, fid);
  const char *p = env->GetStringUTFChars(s, nullptr);
  ans.decoding_method = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(cls, "blankPenalty", "F");
  ans.blank_penalty = env->GetFloatField(config, fid);

  //---------- feat config ----------
  fid = env->GetFieldID(cls, "featConfig",
                        "Lcom/k2fsa/sherpa/ncnn/FeatureConfig;");
  jobject feat_config = env->GetObjectField(config, fid);
  jclass feat_config_cls = env->GetObjectClass(feat_config);

  fid = env->GetFieldID(feat_config_cls, "sampleRate", "I");
  ans.feat_config.sampling_rate = env->GetIntField(feat_config, fid);

  fid = env->GetFieldID(feat_config_cls, "featureDim", "I");
  ans.feat_config.feature_dim = env->GetIntField(feat_config, fid);

  fid = env->GetFieldID(feat_config_cls, "dither", "F");
  ans.feat_config.dither = env->GetFloatField(feat_config, fid);

  //---------- model config ----------
  fid = env->GetFieldID(cls, "modelConfig",
                        "Lcom/k2fsa/sherpa/ncnn/OfflineModelConfig;");
  jobject model_config = env->GetObjectField(config, fid);
  jclass model_config_cls = env->GetObjectClass(model_config);

  fid = env->GetFieldID(model_config_cls, "tokens", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(model_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.tokens = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(model_config_cls, "numThreads", "I");
  ans.model_config.num_threads = env->GetIntField(model_config, fid);

  fid = env->GetFieldID(model_config_cls, "debug", "Z");
  ans.model_config.debug = env->GetBooleanField(model_config, fid);

  // sense voice
  fid = env->GetFieldID(model_config_cls, "senseVoice",
                        "Lcom/k2fsa/sherpa/ncnn/OfflineSenseVoiceModelConfig;");
  jobject sense_voice_config = env->GetObjectField(model_config, fid);
  jclass sense_voice_config_cls = env->GetObjectClass(sense_voice_config);

  fid =
      env->GetFieldID(sense_voice_config_cls, "modelDir", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(sense_voice_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.sense_voice.model_dir = p;
  env->ReleaseStringUTFChars(s, p);

  fid =
      env->GetFieldID(sense_voice_config_cls, "language", "Ljava/lang/String;");
  s = (jstring)env->GetObjectField(sense_voice_config, fid);
  p = env->GetStringUTFChars(s, nullptr);
  ans.model_config.sense_voice.language = p;
  env->ReleaseStringUTFChars(s, p);

  fid = env->GetFieldID(sense_voice_config_cls, "useInverseTextNormalization",
                        "Z");
  ans.model_config.sense_voice.use_itn =
      env->GetBooleanField(sense_voice_config, fid);

  return ans;
}

}  // namespace sherpa_ncnn

SHERPA_NCNN_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_ncnn_OfflineRecognizer_newFromAsset(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jobject asset_manager,
                                                          jobject _config) {
#if __ANDROID_API__ >= 9
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  if (!mgr) {
    SHERPA_NCNN_LOGE("Failed to get asset manager: %p", mgr);
    return 0;
  }
#endif
  auto config = sherpa_ncnn::GetOfflineConfig(env, _config);

  if (config.model_config.debug) {
    // logcat truncates long strings, so we split the string into chunks
    auto str_vec = sherpa_ncnn::SplitString(config.ToString(), 128);
    for (const auto &s : str_vec) {
      SHERPA_NCNN_LOGE("%s", s.c_str());
    }
  }

  auto model = new sherpa_ncnn::OfflineRecognizer(
#if __ANDROID_API__ >= 9
      mgr,
#endif
      config);

  return (jlong)model;
}

SHERPA_NCNN_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_ncnn_OfflineRecognizer_newFromFile(JNIEnv *env,
                                                         jobject /*obj*/,
                                                         jobject _config) {
  auto config = sherpa_ncnn::GetOfflineConfig(env, _config);

  if (config.model_config.debug) {
    auto str_vec = sherpa_ncnn::SplitString(config.ToString(), 128);
    for (const auto &s : str_vec) {
      SHERPA_NCNN_LOGE("%s", s.c_str());
    }
  }

  if (!config.Validate()) {
    SHERPA_NCNN_LOGE("Errors found in config!");
    return 0;
  }

  auto model = new sherpa_ncnn::OfflineRecognizer(config);

  return (jlong)model;
}

SHERPA_NCNN_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_OfflineRecognizer_setConfig(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jobject _config) {
  auto config = sherpa_ncnn::GetOfflineConfig(env, _config);

  if (config.model_config.debug) {
    SHERPA_NCNN_LOGE("set config:\n%s", config.ToString().c_str());
  }

  auto recognizer = reinterpret_cast<sherpa_ncnn::OfflineRecognizer *>(ptr);
  recognizer->SetConfig(config);
}

SHERPA_NCNN_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_OfflineRecognizer_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_ncnn::OfflineRecognizer *>(ptr);
}

SHERPA_NCNN_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_ncnn_OfflineRecognizer_createStream(JNIEnv * /*env*/,
                                                          jobject /*obj*/,
                                                          jlong ptr) {
  auto recognizer = reinterpret_cast<sherpa_ncnn::OfflineRecognizer *>(ptr);
  std::unique_ptr<sherpa_ncnn::OfflineStream> s = recognizer->CreateStream();

  // The user is responsible to free the returned pointer.
  //
  // See Java_com_k2fsa_sherpa_ncnn_OfflineStream_delete() from
  // ./offline-stream.cc
  sherpa_ncnn::OfflineStream *p = s.release();
  return (jlong)p;
}

SHERPA_NCNN_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_OfflineRecognizer_decode(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlong stream_ptr) {
  SafeJNI(env, "OfflineRecognizer_decode", [&] {
    if (!ValidatePointer(env, ptr, "OfflineRecognizer_decode",
                         "OfflineRecognizer pointer is null.") ||
        !ValidatePointer(env, stream_ptr, "OfflineRecognizer_decode",
                         "OfflineStream pointer is null.")) {
      return;
    }

    auto recognizer = reinterpret_cast<sherpa_ncnn::OfflineRecognizer *>(ptr);
    auto stream = reinterpret_cast<sherpa_ncnn::OfflineStream *>(stream_ptr);
    recognizer->DecodeStream(stream);
  });
}

SHERPA_NCNN_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_ncnn_OfflineRecognizer_decodeStreams(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jlongArray stream_ptrs) {
  SafeJNI(env, "OfflineRecognizer_decode_streams", [&] {
    if (!ValidatePointer(env, ptr, "OfflineRecognizer_decode_streams",
                         "OfflineRecognizer pointer is null.")) {
      return;
    }

    auto recognizer = reinterpret_cast<sherpa_ncnn::OfflineRecognizer *>(ptr);

    jlong *p = env->GetLongArrayElements(stream_ptrs, nullptr);
    jsize n = env->GetArrayLength(stream_ptrs);

    auto ss = reinterpret_cast<sherpa_ncnn::OfflineStream **>(p);
    recognizer->DecodeStreams(ss, n);

    env->ReleaseLongArrayElements(stream_ptrs, p, JNI_ABORT);
  });
}

SHERPA_NCNN_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_ncnn_OfflineRecognizer_getResult(JNIEnv *env,
                                                       jobject /*obj*/,
                                                       jlong streamPtr) {
  auto stream = reinterpret_cast<sherpa_ncnn::OfflineStream *>(streamPtr);
  sherpa_ncnn::OfflineRecognizerResult result = stream->GetResult();

  // [0]: text, jstring
  // [1]: tokens, array of jstring
  // [2]: timestamps, array of float
  // [3]: lang, jstring
  // [4]: emotion, jstring
  // [5]: event, jstring
  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      6, env->FindClass("java/lang/Object"), nullptr);

  jstring text = env->NewStringUTF(result.text.c_str());
  env->SetObjectArrayElement(obj_arr, 0, text);

  jobjectArray tokens_arr = (jobjectArray)env->NewObjectArray(
      result.tokens.size(), env->FindClass("java/lang/String"), nullptr);

  int32_t i = 0;
  for (const auto &t : result.tokens) {
    jstring jtext = env->NewStringUTF(t.c_str());
    env->SetObjectArrayElement(tokens_arr, i, jtext);
    i += 1;
  }

  env->SetObjectArrayElement(obj_arr, 1, tokens_arr);

  jfloatArray timestamps_arr = env->NewFloatArray(result.timestamps.size());
  env->SetFloatArrayRegion(timestamps_arr, 0, result.timestamps.size(),
                           result.timestamps.data());

  env->SetObjectArrayElement(obj_arr, 2, timestamps_arr);

  // [3]: lang, jstring
  // [4]: emotion, jstring
  // [5]: event, jstring
  env->SetObjectArrayElement(obj_arr, 3,
                             env->NewStringUTF(result.lang.c_str()));
  env->SetObjectArrayElement(obj_arr, 4,
                             env->NewStringUTF(result.emotion.c_str()));
  env->SetObjectArrayElement(obj_arr, 5,
                             env->NewStringUTF(result.event.c_str()));

  return obj_arr;
}
