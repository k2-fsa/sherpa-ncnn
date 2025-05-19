// sherpa-ncnn/csrc/text-utils.h
//
// Copyright 2009-2011  Saarland University;  Microsoft Corporation
// Copyright      2023  Xiaomi Corporation
#ifndef SHERPA_NCNN_CSRC_TEXT_UTILS_H_
#define SHERPA_NCNN_CSRC_TEXT_UTILS_H_

#include <errno.h>
#include <stdlib.h>

#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _MSC_VER
#define SHERPA_NCNN_STRTOLL(cur_cstr, end_cstr) \
  _strtoi64(cur_cstr, end_cstr, 10);
#else
#define SHERPA_NCNN_STRTOLL(cur_cstr, end_cstr) strtoll(cur_cstr, end_cstr, 10);
#endif

// This file is copied/modified from
// https://github.com/kaldi-asr/kaldi/blob/master/src/util/text-utils.h
namespace sherpa_ncnn {

/// Converts a string into an integer via strtoll and returns false if there was
/// any kind of problem (i.e. the string was not an integer or contained extra
/// non-whitespace junk, or the integer was too large to fit into the type it is
/// being converted into).  Only sets *out if everything was OK and it returns
/// true.
template <class Int>
bool ConvertStringToInteger(const std::string &str, Int *out) {
  // copied from kaldi/src/util/text-util.h
  static_assert(std::is_integral<Int>::value, "");
  const char *this_str = str.c_str();
  char *end = nullptr;
  errno = 0;
  int64_t i = SHERPA_NCNN_STRTOLL(this_str, &end);
  if (end != this_str) {
    while (isspace(*end)) ++end;
  }
  if (end == this_str || *end != '\0' || errno != 0) return false;
  Int iInt = static_cast<Int>(i);
  if (static_cast<int64_t>(iInt) != i ||
      (i < 0 && !std::numeric_limits<Int>::is_signed)) {
    return false;
  }
  *out = iInt;
  return true;
}

// This is defined for F = float and double.
template <typename T>
bool ConvertStringToReal(const std::string &str, T *out);

}  // namespace sherpa_ncnn
#endif  // SHERPA_NCNN_CSRC_TEXT_UTILS_H_
