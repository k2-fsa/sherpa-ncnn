/**
 * Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)
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
#include <algorithm>
#include <memory>

#include "sherpa-ncnn/c-api/c-api.h"

// see also
// https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html

extern "C" {

static_assert(sizeof(SherpaNcnnFeatureExtractorConfig) == 4 * 2, "");
static_assert(sizeof(SherpaNcnnModelConfig) == 4 * 9, "");
static_assert(sizeof(SherpaNcnnDecoderConfig) == 4 * 2, "");
static_assert(sizeof(SherpaNcnnRecognizerConfig) ==
                  4 * 2 + 4 * 9 + 4 * 2 + 4 * 4 + 4 * 2,
              "");

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
