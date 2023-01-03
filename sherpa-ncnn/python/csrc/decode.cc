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

#include "sherpa-ncnn/python/csrc/decode.h"

#include "sherpa-ncnn/csrc/decode.h"
#include "sherpa-ncnn/csrc/model.h"
#include "sherpa-ncnn/python/csrc/mat-util.h"

namespace sherpa_ncnn {

static void PybindGreedySearch(py::module *m) {
  m->def(
      "greedy_search",
      [](Model *model, py::array _encoder_out, py::array _decoder_out,
         std::vector<int32_t> hyp)
          -> std::pair<py::array, std::vector<int32_t>> {
        ncnn::Mat encoder_out = ArrayToMat(_encoder_out);
        ncnn::Mat decoder_out = ArrayToMat(_decoder_out);

        GreedySearch(model, encoder_out, &decoder_out, &hyp);

        return {MatToArray(decoder_out), hyp};
      },
      py::arg("model"), py::arg("encoder_out"), py::arg("decoder_out"),
      py::arg("hyp"));
}

void PybindDecode(py::module *m) { PybindGreedySearch(m); }

}  // namespace sherpa_ncnn
