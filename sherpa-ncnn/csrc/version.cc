// sherpa-ncnn/csrc/version.h
//
// Copyright      2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/version.h"

namespace sherpa_ncnn {

const char *GetGitDate() {
  static const char *date = "Wed Sep 3 16:37:56 2025";
  return date;
}

const char *GetGitSha1() {
  static const char *sha1 = "a278219e";
  return sha1;
}

const char *GetVersionStr() {
  static const char *version = "2.1.12";
  return version;
}

}  // namespace sherpa_ncnn
