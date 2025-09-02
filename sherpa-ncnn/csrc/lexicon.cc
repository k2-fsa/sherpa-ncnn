// sherpa-ncnn/csrc/lexicon.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-ncnn/csrc/lexicon.h"

#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "sherpa-ncnn/csrc/file-utils.h"
#include "sherpa-ncnn/csrc/macros.h"
#include "sherpa-ncnn/csrc/text-utils.h"

namespace sherpa_ncnn {

static std::vector<int32_t> ConvertTokensToIds(
    const std::unordered_map<std::string, int32_t> &token2id,
    const std::vector<std::string> &tokens) {
  std::vector<int32_t> ids;
  ids.reserve(tokens.size());
  for (const auto &s : tokens) {
    if (!token2id.count(s)) {
      SHERPA_NCNN_LOGE("Unknown token: %s. Skip it", s.c_str());
      continue;
    }
    int32_t id = token2id.at(s);
    ids.push_back(id);
  }

  return ids;
}

class Lexicon::Impl {
 public:
  explicit Impl(const std::string &lexicon,
                const std::unordered_map<std::string, int32_t> &token2id)
      : token2id_(token2id) {
    std::ifstream is(lexicon);
    Init(is);
  }

  void TokenizeWord(const std::string &word,
                    std::vector<int32_t> *token_ids) const {
    token_ids->clear();

    auto w = ToLowerCase(word);

    if (word2token_ids_.count(w)) {
      *token_ids = word2token_ids_.at(w);
    }
  }

  void AddWord(const std::string &word, const std::vector<int32_t> &token_ids) {
    auto w = ToLowerCase(word);
    word2token_ids_[w] = token_ids;
  }

 private:
  void Init(std::istream &is) {
    std::string word;
    std::vector<std::string> token_list;
    std::string line;
    std::string token;

    while (std::getline(is, line)) {
      std::istringstream iss(line);

      token_list.clear();

      iss >> word;
      ToLowerCase(&word);

      if (word2token_ids_.count(word)) {
        SHERPA_NCNN_LOGE("Duplicated word: %s. Ignore it.", word.c_str());
        continue;
      }

      while (iss >> token) {
        token_list.push_back(std::move(token));
      }

      std::vector<int32_t> ids = ConvertTokensToIds(token2id_, token_list);
      if (ids.empty()) {
        SHERPA_NCNN_LOGE("Empty token IDs for word %s. Line: %s", word.c_str(),
                         line.c_str());
        continue;
      }
      static int32_t count = 0;
      count++;
      if (count < 10) {
        std::ostringstream os;
        os << word << " -> ";
        for (auto t : token_list) {
          os << t << " ";
        }
        os << "\n";
        SHERPA_NCNN_LOGE("%s", os.str().c_str());
      }

      word2token_ids_.insert({std::move(word), std::move(ids)});
    }
  }

 private:
  std::unordered_map<std::string, std::vector<int32_t>> word2token_ids_;
  std::unordered_map<std::string, int32_t> token2id_;
};

Lexicon::~Lexicon() = default;

Lexicon::Lexicon(const std::string &lexicon,
                 const std::unordered_map<std::string, int32_t> &token2id)
    : impl_(std::make_unique<Impl>(lexicon, token2id)) {}

void Lexicon::TokenizeWord(const std::string &word,
                           std::vector<int32_t> *token_ids) const {
  impl_->TokenizeWord(word, token_ids);
}

void Lexicon::AddWord(const std::string &word,
                      const std::vector<int32_t> &token_ids) const {
  impl_->AddWord(word, token_ids);
}

}  // namespace sherpa_ncnn
