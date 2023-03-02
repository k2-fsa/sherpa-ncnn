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
#include <utility>
#include <vector>

#include "net.h"  // NOLINT
#include "sherpa-ncnn/csrc/model.h"

namespace sherpa_ncnn {

class LstmModel : public Model {
 public:
  explicit LstmModel(const ModelConfig &config);
#if __ANDROID_API__ >= 9
  LstmModel(AAssetManager *mgr, const ModelConfig &config);
#endif

  ncnn::Net &GetEncoder() override { return encoder_; }
  ncnn::Net &GetDecoder() override { return decoder_; }
  ncnn::Net &GetJoiner() override { return joiner_; }

  std::vector<ncnn::Mat> GetEncoderInitStates() const override;

  /** Run the encoder network.
   *
   * @param features  A 2-d mat of shape (num_frames, feature_dim).
   *                  Note: features.w = feature_dim.
   *                        features.h = num_frames.
   * @param states Contains two tensors:
   *          - hx  Hidden state of the LSTM model. You can leave it to empty
   *                on the first invocation. It is changed in-place.
   *
   *          - cx  Hidden cell state of the LSTM model. You can leave it to
   *                empty on the first invocation. It is changed in-place.
   *
   *          - Note: on the first invocation, you can pass an empty vector.
   *
   * @return Return a pair containing:
   *   - the output of the encoder. Its shape is (num_out_frames, encoder_dim).
   *     Note: ans.w == encoder_dim; ans.h == num_out_frames
   *
   *   - next_states, a vector containing hx and cx for the next invocation
   */
  std::pair<ncnn::Mat, std::vector<ncnn::Mat>> RunEncoder(
      ncnn::Mat &features, const std::vector<ncnn::Mat> &states) override;

  std::pair<ncnn::Mat, std::vector<ncnn::Mat>> RunEncoder(
      ncnn::Mat &features, const std::vector<ncnn::Mat> &states,
      ncnn::Extractor *extractor) override;

  ncnn::Mat RunDecoder(ncnn::Mat &decoder_input) override;

  ncnn::Mat RunDecoder(ncnn::Mat &decoder_input,
                       ncnn::Extractor *extractor) override;

  ncnn::Mat RunJoiner(ncnn::Mat &encoder_out, ncnn::Mat &decoder_out) override;

  ncnn::Mat RunJoiner(ncnn::Mat &encoder_out, ncnn::Mat &decoder_out,
                      ncnn::Extractor *extractor) override;

  int32_t Segment() const override { return 9; }

  // Advance the feature extract by this number of frames after
  // running the encoder network
  int32_t Offset() const override { return 4; }

 private:
  void InitEncoder(const std::string &encoder_param,
                   const std::string &encoder_bin);
  void InitDecoder(const std::string &decoder_param,
                   const std::string &decoder_bin);
  void InitJoiner(const std::string &joiner_param,
                  const std::string &joiner_bin);

  void InitEncoderPostProcessing();

#if __ANDROID_API__ >= 9
  void InitEncoder(AAssetManager *mgr, const std::string &encoder_param,
                   const std::string &encoder_bin);
  void InitDecoder(AAssetManager *mgr, const std::string &decoder_param,
                   const std::string &decoder_bin);
  void InitJoiner(AAssetManager *mgr, const std::string &joiner_param,
                  const std::string &joiner_bin);
#endif

  void InitEncoderInputOutputIndexes();
  void InitDecoderInputOutputIndexes();
  void InitJoinerInputOutputIndexes();

 private:
  // arg0: 3
  int32_t num_encoder_layers_ = 12;  // arg1
  int32_t encoder_dim_ = 512;        // arg2, i.e., d_model
  int32_t rnn_hidden_size_ = 1024;   // arg3

  ncnn::Net encoder_;
  ncnn::Net decoder_;
  ncnn::Net joiner_;

  std::vector<int32_t> encoder_input_indexes_;
  std::vector<int32_t> encoder_output_indexes_;

  std::vector<int32_t> decoder_input_indexes_;
  std::vector<int32_t> decoder_output_indexes_;

  std::vector<int32_t> joiner_input_indexes_;
  std::vector<int32_t> joiner_output_indexes_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_LSTM_MODEL_H_
