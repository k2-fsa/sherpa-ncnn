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

#ifndef SHERPA_NCNN_CSRC_MODEL_H_
#define SHERPA_NCNN_CSRC_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "net.h"  // NOLINT

namespace sherpa_ncnn {

struct ModelConfig {
  std::string encoder_param;  // path to encoder.ncnn.param
  std::string encoder_bin;    // path to encoder.ncnn.bin
  std::string decoder_param;  // path to decoder.ncnn.param
  std::string decoder_bin;    // path to decoder.ncnn.bin
  std::string joiner_param;   // path to joiner.ncnn.param
  std::string joiner_bin;     // path to joiner.ncnn.bin
  int32_t num_threads;        // number of threads to run the model
  std::string ToString() const;
};

class Model {
 public:
  virtual ~Model() = default;

  /** Create a model from a config. */
  static std::unique_ptr<Model> Create(const ModelConfig &config);

  static void InitNet(ncnn::Net &net, const std::string &param,
                      const std::string &bin);

  /** Run the encoder network.
   *
   * @param features  A 2-d mat of shape (num_frames, feature_dim).
   *                  Note: features.w = feature_dim.
   *                        features.h = num_frames.
   * @param states It contains the states for the encoder network. Its exact
   *               content is determined by the underlying network.
   *
   * @return Return a pair containing:
   *   - encoder_out
   *   - next_states
   */
  virtual std::pair<ncnn::Mat, std::vector<ncnn::Mat>> RunEncoder(
      ncnn::Mat &features, const std::vector<ncnn::Mat> &states) = 0;

  /** Run the decoder network.
   *
   * @param  decoder_input A mat of shape (context_size,). Note: Its underlying
   *                       content consists of integers, though its type is
   *                       float.
   *
   * @return Return a mat of shape (decoder_dim,)
   */
  virtual ncnn::Mat RunDecoder(ncnn::Mat &decoder_input) = 0;

  /** Run the joiner network.
   *
   * @param encoder_out  A mat of shape (encoder_dim,)
   * @param decoder_out  A mat of shape (decoder_dim,)
   *
   * @return Return the joiner output which is of shape (vocab_size,)
   */
  virtual ncnn::Mat RunJoiner(ncnn::Mat &encoder_out,
                              ncnn::Mat &decoder_out) = 0;

  virtual int32_t ContextSize() const { return 2; }

  virtual int32_t BlankId() const { return 0; }

  // The encoder takes this number of frames as input
  virtual int32_t Segment() const = 0;

  // Advance the feature extractor by this number of frames after
  // running the encoder network
  virtual int32_t Offset() const = 0;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_MODEL_H_
