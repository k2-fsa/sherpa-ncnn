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

#ifndef SHERPA_NCNN_CSRC_DECODE_H_
#define SHERPA_NCNN_CSRC_DECODE_H_

#include <vector>

#include "net.h"  // NOLINT
#include "sherpa-ncnn/csrc/model.h"

namespace sherpa_ncnn {

/**
 *
 * @param model  The neural network.
 * @param encoder_out  Its shape is (num_frames, encoder_out_dim).
 *                     encoder_out.w == encoder_out_dim
 *                     encoder_out.h == num_frames
 * @param decoder_out  Its shape is (1, decoder_out_dim).
 *                     decoder_out.w == decoder_out_dim
 *                     decoder_out.h == 1
 * @param hyp The recognition result. It is changed in place.
 */
void GreedySearch(Model *model, ncnn::Mat &encoder_out, ncnn::Mat *decoder_out,
                  std::vector<int32_t> *hyp);

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_DECODE_H_
