// sherpa-ncnn/csrc/math.cc
//
// Copyright 2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/math.h"

#include <random>

namespace sherpa_ncnn {

void RandomVectorFill(float *p, int32_t n, float a /*= 0*/, float b /*= 1*/) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(a, b);

  for (int32_t i = 0; i < n; ++i) {
    p[i] = dist(gen);
  }
}

}  // namespace sherpa_ncnn
