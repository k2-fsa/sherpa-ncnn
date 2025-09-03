// sherpa-ncnn/csrc/sherpa-ncnn-version.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <stdio.h>

#include <cstdint>

#include "sherpa-ncnn/csrc/version.h"

int32_t main() {
  printf("sherpa-ncnn version : %s\n", sherpa_ncnn::GetVersionStr());
  printf("sherpa-ncnn Git SHA1: %s\n", sherpa_ncnn::GetGitSha1());
  printf("sherpa-ncnn Git date: %s\n", sherpa_ncnn::GetGitDate());

  return 0;
}
