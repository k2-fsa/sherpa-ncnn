// sherpa-ncnn/csrc/version.cc
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/version.h"

namespace sherpa_ncnn {

const char *GetGitDate() {
  static const char *date = "Fri Sep 5 23:27:34 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "3c4dbcbf";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "2.1.14";
  return version;
}

}  // namespace sherpa_ncnn
