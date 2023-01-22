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

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Please refer to
/// https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
/// to download pre-trained models. That is, you can find .ncnn.param,
/// .ncnn.bin, and tokens.txt for this struct from there.
struct SherpaNcnnModelConfig {
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
};

struct SherpaNcnnDecoderConfig {
  /// Decoding method. Supported values are:
  /// greedy_search, modified_beam_search
  const char *decoding_method;

  /// Number of active paths for modified_beam_search.
  /// It is ignored when decoding_method is greedy_search.
  int32_t num_active_paths;

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
};

struct SherpaNcnnResult {
  const char *text;
  // TODO: Add more fields
};

typedef struct SherpaNcnnRecognizer SherpaNcnnRecognizer;

/// Create a recognizer.
///
/// @param model_config  Config for the model.
/// @param decoder_config Config for decoding.
/// @return Return a pointer to the recognizer. The user has to invoke
//          DestroyRecognizer() to free it to avoid memory leak.
SherpaNcnnRecognizer *CreateRecognizer(
    const SherpaNcnnModelConfig *model_config,
    const SherpaNcnnDecoderConfig *decoder_config);

/// Free a pointer returned by CreateRecognizer()
///
/// @param p A pointer returned by CreateRecognizer()
void DestroyRecognizer(SherpaNcnnRecognizer *p);

/// Accept input audio samples and compute the features.
/// The user has to invoke Decode() to run the neural network and decoding.
///
/// @param p  A pointer returned by CreateRecognizer().
/// @param sample_rate  Sampler ate of the input samples. It has to be 16 kHz
///                     for models from icefall.
/// @param samples A pointer to a 1-D array containing audio samples.
///                The range of samples has to be normalized to [-1, 1].
/// @param n  Number of elements in the samples array.
void AcceptWaveform(SherpaNcnnRecognizer *p, float sample_rate,
                    const float *samples, int32_t n);

/// If there are enough number of feature frames, it invokes the neural network
/// computation and decoding. Otherwise, it is a no-op.
void Decode(SherpaNcnnRecognizer *p);

/// Get the decoding results so far.
///
/// @param p A pointer returned by CreateRecognizer().
/// @return A pointer containing the result. The user has to invoke
///         DestroyResult() to free the returned pointer to avoid memory leak.
SherpaNcnnResult *GetResult(SherpaNcnnRecognizer *p);

/// Destroy the pointer returned by GetResult().
///
/// @param r A pointer returned by GetResult()
void DestroyResult(const SherpaNcnnResult *r);

/// Reset the recognizer, which clears the neural network model state
/// and the state for decoding.
///
/// @param p A pointer returned by CreateRecognizer().
void Reset(SherpaNcnnRecognizer *p);

/// Signal that no more audio samples would be available.
/// After this call, you cannot call AcceptWaveform() any more.
///
/// @param p A pointer returned by CreateRecognizer()
void InputFinished(SherpaNcnnRecognizer *p);

/// Return 1 is an endpoint has been detected.
///
/// Caution: You have to call this function before GetResult().
///
/// @param p A pointer returned by CreateRecognizer()
/// @return Return 1 if an endpoint is detected. Return 0 otherwise.
int32_t IsEndpoint(SherpaNcnnRecognizer *p);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // SHERPA_NCNN_C_API_C_API_H_
