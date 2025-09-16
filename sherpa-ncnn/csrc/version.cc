// sherpa-ncnn/csrc/version.cc
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/version.h"

namespace sherpa_ncnn {

const char *GetGitDate() {
  static const char *date = "Tue Sep 16 15:57:04 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "d4bb5a78";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "2.1.15";
  return version;
}

}  // namespace sherpa_ncnn
