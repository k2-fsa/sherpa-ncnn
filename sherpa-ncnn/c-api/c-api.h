/**
 * Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
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

#ifndef SHERPA_NCNN_C_API_C_API_H_
#define SHERPA_NCNN_C_API_C_API_H_

// C API for sherpa-ncnn
//
// Please refer to
// https://github.com/k2-fsa/sherpa-ncnn/blob/master/c-api-examples/decode-file-c-api.c
// for usages.

#include <stdint.h>


#if defined(__BORLANDC__)
    #if defined(__clang__)
        #ifdef __cplusplus
           #define SherpaNcnn_EXPORT extern "C"
        #else
           #define SherpaNcnn_EXPORT
        #endif
    #else
        #error Cannot define PACKED macros for this compiler
    #endif
#elif defined(_MSC_VER)
    #ifdef __cplusplus
           #define SherpaNcnn_EXPORT   extern "C" _declspec(dllexport)
    #else
       #define SherpaNcnn_EXPORT
   #endif

#elif defined(__GNUC__)
    #ifdef __cplusplus
          #define SherpaNcnn_EXPORT extern "C"
    #else
          #define SherpaNcnn_EXPORT
    #endif
#else
    #error PACKED macros are not defined for this compiler
#endif


#ifdef __cplusplus
extern "C" {
#endif

/// Please refer to
/// https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
/// to download pre-trained models. That is, you can find .ncnn.param,
/// .ncnn.bin, and tokens.txt for this struct from there.
SherpaNcnn_EXPORT typedef struct SherpaNcnnModelConfig {
  /// Path to encoder.ncnn.param
  const char *encoder_param;

  /// Path to encoder.ncnn.bin
  const char *encoder_bin;

  /// Path to decoder.ncnn.param
  const char *decoder_param;

  /// Path to decoder.ncnn.bin
  const char *decoder_bin;

  /// Path to joiner.ncnn.param
  const char *joiner_param;

  /// Path to joiner.ncnn.bin
  const char *joiner_bin;

  /// Path to tokens.txt
  const char *tokens;

  /// If it is non-zero, and it has GPU available, and ncnn is built
  /// with vulkan, then it will use GPU for computation.
  /// Otherwise, it uses CPU for computation.
  int32_t use_vulkan_compute;

  /// Number of threads for neural network computation.
  int32_t num_threads;
} SherpaNcnnModelConfig;

SherpaNcnn_EXPORT typedef struct SherpaNcnnDecoderConfig {
  /// Decoding method. Supported values are:
  /// greedy_search, modified_beam_search
  const char *decoding_method;

  /// Number of active paths for modified_beam_search.
  /// It is ignored when decoding_method is greedy_search.
  int32_t num_active_paths;
} SherpaNcnnDecoderConfig;

SherpaNcnn_EXPORT typedef struct SherpaNcnnFeatureExtractorConfig {
  // Sampling rate of the input audio samples. MUST match the one
  // expected by the model. For instance, it should be 16000 for models
  // from icefall.
  float sampling_rate;

  // feature dimension. Must match the one expected by the model.
  // For instance, it should be 80 for models from icefall.
  int32_t feature_dim;
} SherpaNcnnFeatureExtractorConfig;

SherpaNcnn_EXPORT typedef struct SherpaNcnnRecognizerConfig {
  SherpaNcnnFeatureExtractorConfig feat_config;
  SherpaNcnnModelConfig model_config;
  SherpaNcnnDecoderConfig decoder_config;

  /// 0 to disable endpoint detection.
  /// A non-zero value to enable endpoint detection.
  int32_t enable_endpoint;

  /// An endpoint is detected if trailing silence in seconds is larger than
  /// this value even if nothing has been decoded.
  /// Used only when enable_endpoint is not 0.
  float rule1_min_trailing_silence;

  /// An endpoint is detected if trailing silence in seconds is larger than
  /// this value after something that is not blank has been decoded.
  /// Used only when enable_endpoint is not 0.
  float rule2_min_trailing_silence;

  /// An endpoint is detected if the utterance in seconds is larger than
  /// this value.
  /// Used only when enable_endpoint is not 0.
  float rule3_min_utterance_length;
} SherpaNcnnRecognizerConfig;

SherpaNcnn_EXPORT typedef struct SherpaNcnnResult {
  // Recognized text
  const char *text;

  // Pointer to continuous memory which holds string based tokens
  // which are seperated by \0
  const char *tokens;

  // Pointer to continuous memory which holds timestamps which
  // are seperated by \0
  float* timestamps;

  // The number of tokens/timestamps in above pointer
  int32_t count;
} SherpaNcnnResult;

SherpaNcnn_EXPORT typedef struct SherpaNcnnRecognizer SherpaNcnnRecognizer;
SherpaNcnn_EXPORT typedef struct SherpaNcnnStream SherpaNcnnStream;

/// Create a recognizer.
///
/// @param config  Config for the recognizer.
/// @return Return a pointer to the recognizer. The user has to invoke
//          DestroyRecognizer() to free it to avoid memory leak.
SherpaNcnn_EXPORT SherpaNcnnRecognizer *CreateRecognizer(
    const SherpaNcnnRecognizerConfig *config);

/// Free a pointer returned by CreateRecognizer()
///
/// @param p A pointer returned by CreateRecognizer()
SherpaNcnn_EXPORT void DestroyRecognizer(SherpaNcnnRecognizer *p);

/// Create a stream for accepting audio samples
///
/// @param p A pointer returned by CreateRecognizer
/// @return Return a pointer to a stream. The caller MUST invoke
///         DestroyStream at the end to avoid memory leak.
SherpaNcnn_EXPORT SherpaNcnnStream *CreateStream(SherpaNcnnRecognizer *p);

SherpaNcnn_EXPORT void DestroyStream(SherpaNcnnStream *s);

/// Accept input audio samples and compute the features.
///
/// @param s  A pointer returned by CreateStream().
/// @param sample_rate  Sample rate of the input samples. If it is different
///                     from feat_config.sampling_rate, we will do resample.
///                     Caution: You MUST not use a different sampling_rate
///                     across different calls to AcceptWaveform()
/// @param samples A pointer to a 1-D array containing audio samples.
///                The range of samples has to be normalized to [-1, 1].
/// @param n  Number of elements in the samples array.
SherpaNcnn_EXPORT void AcceptWaveform(SherpaNcnnStream *s, float sample_rate,
                    const float *samples, int32_t n);

/// Test if the stream has enough frames for decoding.
///
/// The common usage is:
///   while (IsReady(p, s)) {
///      Decode(p, s);
///   }
/// @param p A pointer returned by CreateRecognizer()
/// @param s A pointer returned by CreateStream()
/// @return Return 1 if the given stream is ready for decoding.
///         Return 0 otherwise.
SherpaNcnn_EXPORT int32_t IsReady(SherpaNcnnRecognizer *p, SherpaNcnnStream *s);

/// Pre-condition for this function:
///   You must ensure that IsReady(p, s) return 1 before calling this function.
///
/// @param p A pointer returned by CreateRecognizer()
/// @param s A pointer returned by CreateStream()
SherpaNcnn_EXPORT void Decode(SherpaNcnnRecognizer *p, SherpaNcnnStream *s);

/// Get the decoding results so far.
///
/// @param p A pointer returned by CreateRecognizer().
/// @param s A pointer returned by CreateStream()
/// @return A pointer containing the result. The user has to invoke
///         DestroyResult() to free the returned pointer to avoid memory leak.
SherpaNcnn_EXPORT SherpaNcnnResult *GetResult(SherpaNcnnRecognizer *p, SherpaNcnnStream *s);

/// Destroy the pointer returned by GetResult().
///
/// @param r A pointer returned by GetResult()
SherpaNcnn_EXPORT void DestroyResult(const SherpaNcnnResult *r);

/// Reset a stream
///
/// @param p A pointer returned by CreateRecognizer().
/// @param s A pointer returned by CreateStream().
SherpaNcnn_EXPORT void Reset(SherpaNcnnRecognizer *p, SherpaNcnnStream *s);

/// Signal that no more audio samples would be available.
/// After this call, you cannot call AcceptWaveform() any more.
///
/// @param s A pointer returned by CreateStream()
SherpaNcnn_EXPORT void InputFinished(SherpaNcnnStream *s);

/// Return 1 is an endpoint has been detected.
///
/// Common usage:
///   if (IsEndpoint(p, s)) {
///     Reset(p, s);
///   }
///
/// @param p A pointer returned by CreateRecognizer()
/// @return Return 1 if an endpoint is detected. Return 0 otherwise.
SherpaNcnn_EXPORT int32_t IsEndpoint(SherpaNcnnRecognizer *p, SherpaNcnnStream *s);

// for displaying results on Linux/macOS.
SherpaNcnn_EXPORT typedef struct SherpaNcnnDisplay SherpaNcnnDisplay;

/// Create a display object. Must be freed using DestroyDisplay to avoid
/// memory leak.
SherpaNcnn_EXPORT SherpaNcnnDisplay *CreateDisplay(int32_t max_word_per_line);

SherpaNcnn_EXPORT void DestroyDisplay(SherpaNcnnDisplay *display);

/// Print the result.
SherpaNcnn_EXPORT void SherpaNcnnPrint(SherpaNcnnDisplay *display, int32_t idx, const char *s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // SHERPA_NCNN_C_API_C_API_H_
