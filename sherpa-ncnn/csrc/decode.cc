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

#include "sherpa-ncnn/csrc/decode.h"

namespace sherpa_ncnn {

void GreedySearch(Model *model, ncnn::Mat &encoder_out, ncnn::Mat *decoder_out,
                  std::vector<int32_t> *hyp) {
  int32_t context_size = 2;
  int32_t blank_id = 0;  // hard-code it to 0
  ncnn::Mat decoder_input(context_size);

  for (int32_t t = 0; t != encoder_out.h; ++t) {
    ncnn::Mat encoder_out_t(encoder_out.w, encoder_out.row(t));
    ncnn::Mat joiner_out = model->RunJoiner(encoder_out_t, *decoder_out);

    auto y = static_cast<int32_t>(std::distance(
        static_cast<const float *>(joiner_out),
        std::max_element(
            static_cast<const float *>(joiner_out),
            static_cast<const float *>(joiner_out) + joiner_out.w)));

    if (y != blank_id) {
      static_cast<int32_t *>(decoder_input)[0] = hyp->back();
      static_cast<int32_t *>(decoder_input)[1] = y;
      hyp->push_back(y);

      *decoder_out = model->RunDecoder(decoder_input);
    }
  }
}

}  // namespace sherpa_ncnn
