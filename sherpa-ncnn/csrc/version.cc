// sherpa-ncnn/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/version.h"

namespace sherpa_ncnn {

const char *GetGitDate() {
  static const char *date = "Wed Sep 3 23:39:44 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "b0928b0e";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "2.1.13";
  return version;
}

}  // namespace sherpa_ncnn
