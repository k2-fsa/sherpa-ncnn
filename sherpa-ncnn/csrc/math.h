/**
 * Copyright      2022  Xiaomi Corporation (authors: Daniel Povey)
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
// This file is copied from k2/csrc/utils.h
#ifndef SHERPA_NCNN_CSRC_MATH_H_
#define SHERPA_NCNN_CSRC_MATH_H_

#include <cmath>

namespace sherpa_ncnn {

// logf(FLT_EPSILON)
#define SHERPA_NCNN_MIN_LOG_DIFF_FLOAT -15.9423847198486328125f

// log(DBL_EPSILON)
#define SHERPA_NCNN_MIN_LOG_DIFF_DOUBLE \
  -36.0436533891171535515240975655615329742431640625

template <typename T>
struct LogAdd;

template <>
struct LogAdd<double> {
  double operator()(double x, double y) const {
    double diff;

    if (x < y) {
      diff = x - y;
      x = y;
    } else {
      diff = y - x;
    }
    // diff is negative.  x is now the larger one.

    if (diff >= SHERPA_NCNN_MIN_LOG_DIFF_DOUBLE) {
      double res;
      res = x + log1p(exp(diff));
      return res;
    }

    return x;  // return the larger one.
  }
};

template <>
struct LogAdd<float> {
  float operator()(float x, float y) const {
    float diff;

    if (x < y) {
      diff = x - y;
      x = y;
    } else {
      diff = y - x;
    }
    // diff is negative.  x is now the larger one.

    if (diff >= SHERPA_NCNN_MIN_LOG_DIFF_DOUBLE) {
      float res;
      res = x + log1pf(expf(diff));
      return res;
    }

    return x;  // return the larger one.
  }
};

}  // namespace sherpa_ncnn
#endif  // SHERPA_NCNN_CSRC_MATH_H_
