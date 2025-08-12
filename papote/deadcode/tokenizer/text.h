#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "token_pair.h"

using TokenPairsCount = std::unordered_map<TokenPair, int>;

class Text {
   public:
    Text() = default;
    Text(const std::string& text)
        : text_(reinterpret_cast<const uint8_t*>(text.data()),
                reinterpret_cast<const uint8_t*>(text.data()) + text.size()) {}
    Text(std::vector<int32_t> text) : text_(std::move(text)) {}

    size_t length() const { return text_.size(); }
    size_t size() const { return text_.size(); }
    void save(const std::string& path);
    static Text load(const std::string& path);

    const std::vector<int32_t>& as_tokens() {
        garbage_collect();
        return text_;
    }

    void set_tokens(const std::vector<int32_t>& tokens) {
        text_ = tokens;
        dirty_ = 0;
    }

    void garbage_collect() {
        if (dirty_ == 0) {
            return;
        }

        text_.erase(std::remove(text_.begin(), text_.end(), -1), text_.end());

        dirty_ = 0;
    }

    int merge(int t1, int t2, int new_token);

    std::vector<std::string> as_str_tokens(
        const std::vector<std::string>& vocab) {
        garbage_collect();
        std::vector<std::string> out;
        for (auto t : text_) {
            out.push_back(vocab[t]);
        }
        return out;
    }

    std::string as_str(const std::vector<std::string>& vocab,
                       std::string* sep = nullptr) const {
        std::string result;
        for (auto t : text_) {
            if (t == -1) {
                continue;
            }
            result += vocab[t];
            if (sep != nullptr) {
                std::cout << "sep: " << *sep << std::endl;
                result += *sep;
            }
        }
        return result;
    }

    TokenPairsCount most_frequent_pair() const;

    void unicode_private_to_token();
    void tokenize(const std::vector<TokenPair>& merges);
    void fast_tokenize(const std::vector<TokenPair>& merges,
                       const std::vector<int>* nonterminals = nullptr);

   private:
    static inline size_t skip(size_t i, const std::vector<int>& t) {
        while (i < t.size() && t[i] == -1) {
            ++i;
        }
        return i;
    }
    std::vector<int32_t> text_;
    int dirty_ = 0;
};
