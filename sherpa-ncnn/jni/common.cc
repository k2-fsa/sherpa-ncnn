// sherpa-ncnn/jni/common.cc
//
// Copyright (c)  2025  Xiaomi Corporation
#include "sherpa-ncnn/jni/common.h"

// see
// https://stackoverflow.com/questions/29043872/android-jni-return-multiple-variables
jobject NewInteger(JNIEnv *env, int32_t value) {
  jclass cls = env->FindClass("java/lang/Integer");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(I)V");
  return env->NewObject(cls, constructor, value);
}

jobject NewFloat(JNIEnv *env, float value) {
  jclass cls = env->FindClass("java/lang/Float");
  jmethodID constructor = env->GetMethodID(cls, "<init>", "(F)V");
  return env->NewObject(cls, constructor, value);
}
