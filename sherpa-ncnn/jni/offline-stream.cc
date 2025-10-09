// sherpa-ncnn/jni/offline-stream.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/offline-stream.h"

#include "sherpa-ncnn/jni/common.h"

SHERPA_NCNN_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_OfflineStream_delete(
    JNIEnv * /*env*/, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_ncnn::OfflineStream *>(ptr);
}

SHERPA_NCNN_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_ncnn_OfflineStream_acceptWaveform(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray samples,
    jint sample_rate) {
  auto stream = reinterpret_cast<sherpa_ncnn::OfflineStream *>(ptr);

  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);
  stream->AcceptWaveform(sample_rate, p, n);
  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
}
