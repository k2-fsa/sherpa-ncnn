// sherpa-ncnn/csrc/lexicon.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_NCNN_CSRC_LEXICON_H_
#define SHERPA_NCNN_CSRC_LEXICON_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sherpa_ncnn {

class Lexicon {
 public:
  ~Lexicon();

  Lexicon(const std::string& lexicon,
          const std::unordered_map<std::string, int32_t>& token2id);

  void TokenizeWord(const std::string& word,
                    std::vector<std::string>* tokens) const;

  void TokenizeWord(const std::string& word,
                    std::vector<int32_t>* token_ids) const;

  // return true if the word is added to the lexicon.
  // return false if it fails to add the word
  bool AddWord(const std::string& word,
               const std::vector<std::string>& tokens) const;

  bool AddWord(const std::string& word,
               const std::vector<int32_t>& token_ids) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_ncnn

#endif  // SHERPA_NCNN_CSRC_LEXICON_H_
