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

#ifndef SHERPA_NCNN_CSRC_LSTM_MODEL_H_
#define SHERPA_NCNN_CSRC_LSTM_MODEL_H_

#include <string>

#include "net.h"  // NOLINT

namespace sherpa_ncnn {

class LstmModel {
 public:
  /**
   * @param encoder_param Path to encoder.ncnn.param
   * @param encoder_bin Path to encoder.ncnn.bin
   * @param decoder_param Path to decoder.ncnn.param
   * @param decoder_bin Path to decoder.ncnn.bin
   * @param joiner_param Path to joiner.ncnn.param
   * @param joiner_bin Path to joiner.ncnn.bin
   * @param num_threads Number of threads to use when running the network
   */
  LstmModel(const std::string &encoder_param, const std::string &encoder_bin,
            const std::string &decoder_param, const std::string &decoder_bin,
            const std::string &joiner_param, const std::string &joiner_bin,
            int32_t num_threads);

  /** Run the encoder network.
   *
   * @param features  A 2-d mat of shape (num_frames, feature_dim).
   *                  Note: features.w = feature_dim.
   *                        features.h = num_frames.
   * @param hx  Hidden state of the LSTM model. You can leave it to empty
   *            on the first invocation. It is changed in-place.
   *
   * @param cx  Hidden cell state of the LSTM model. You can leave it to empty
   *            on the first invocation. It is changed in-place.
   *
   * @return Return the output of the encoder. Its shape is
   *  (num_out_frames, encoder_dim).
   *  Note: ans.w == encoder_dim; ans.h == num_out_frames
   */
  ncnn::Mat RunEncoder(ncnn::Mat &features, ncnn::Mat *hx, ncnn::Mat *cx);

  /** Run the decoder network.
   *
   * @param  decoder_input A mat of shape (context_size,). Note: Its underlying
   *                       content consists of integers, though its type is
   *                       float.
   *
   * @return Return a mat of shape (decoder_dim,)
   */
  ncnn::Mat RunDecoder(ncnn::Mat &decoder_input);

  /** Run the joiner network.
   *
   * @param encoder_out  A mat of shape (encoder_dim,)
   * @param decoder_out  A mat of shape (decoder_dim,)
   *
   * @return Return the joiner output which is of shape (vocab_size,)
   */
  ncnn::Mat RunJoiner(ncnn::Mat &encoder_out, ncnn::Mat &decoder_out);

  int32_t ContextSize() const { return 2; }
  int32_t BlankId() const { return 0; }

 private:
  void InitEncoder(const std::string &encoder_param,
                   const std::string &encoder_bin);
  void InitDecoder(const std::string &decoder_param,
                   const std::string &decoder_bin);
  void InitJoiner(const std::string &joiner_param,
                  const std::string &joiner_bin);

 private:
  ncnn::Net encoder_;
  ncnn::Net decoder_;
  ncnn::Net joiner_;

  int32_t num_threads_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_LSTM_MODEL_H_
