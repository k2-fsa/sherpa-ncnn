// sherpa-ncnn/csrc/macros.h
//
// Copyright      2023-2025  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_MACROS_H_
#define SHERPA_NCNN_CSRC_MACROS_H_
#include <stdio.h>
#include <stdlib.h>

#include <utility>
#if __OHOS__
#include "hilog/log.h"

#undef LOG_DOMAIN
#undef LOG_TAG

// https://gitee.com/openharmony/docs/blob/145a084f0b742e4325915e32f8184817927d1251/en/contribute/OpenHarmony-Log-guide.md#hilog-api-usage-specifications
#define LOG_DOMAIN 0x6666
#define LOG_TAG "sherpa_ncnn"
#endif

#if __ANDROID_API__ >= 8
#include "android/log.h"
#define SHERPA_NCNN_LOGE(...)                                            \
  do {                                                                   \
    fprintf(stderr, "%s:%s:%d ", __FILE__, __func__,                     \
            static_cast<int>(__LINE__));                                 \
    fprintf(stderr, ##__VA_ARGS__);                                      \
    fprintf(stderr, "\n");                                               \
    __android_log_print(ANDROID_LOG_WARN, "sherpa-ncnn", ##__VA_ARGS__); \
  } while (0)
#elif defined(__OHOS__)
#define SHERPA_NCNN_LOGE(...) OH_LOG_INFO(LOG_APP, ##__VA_ARGS__)
#elif SHERPA_NCNN_ENABLE_WASM
#define SHERPA_NCNN_LOGE(...)                        \
  do {                                               \
    fprintf(stdout, "%s:%s:%d ", __FILE__, __func__, \
            static_cast<int>(__LINE__));             \
    fprintf(stdout, ##__VA_ARGS__);                  \
    fprintf(stdout, "\n");                           \
  } while (0)
#else
#define SHERPA_NCNN_LOGE(...)                        \
  do {                                               \
    fprintf(stderr, "%s:%s:%d ", __FILE__, __func__, \
            static_cast<int>(__LINE__));             \
    fprintf(stderr, ##__VA_ARGS__);                  \
    fprintf(stderr, "\n");                           \
  } while (0)
#endif

#define SHERPA_NCNN_EXIT(code) exit(code)

#endif  // SHERPA_NCNN_CSRC_MACROS_H_
