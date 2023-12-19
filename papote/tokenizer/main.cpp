#include <algorithm>
#include <chrono>
#include <codecvt>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <locale>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "text.h"
#include "token_pair.h"

using json = nlohmann::json;

//----------- TokenPair
inline int hash_combine(int lhs, int rhs) {
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
}

namespace std {
int hash<TokenPair>::operator()(const TokenPair& p) const {
    return hash_combine(std::hash<int32_t>()(p.first),
                        std::hash<int32_t>()(p.second));
}
};  // namespace std

void to_json(json& j, const TokenPair& p) { j = json{p.first, p.second}; }
void from_json(const json& j, TokenPair& p) {
    j.at(0).get_to(p.first);
    j.at(1).get_to(p.second);
}

//----------- Text

void Text::save(const std::string& path) {
    garbage_collect();

    std::ofstream file(path, std::ios::binary);
    int size = text_.size();
    file.write(reinterpret_cast<char*>(&size), sizeof(size));
    file.write(reinterpret_cast<char*>(text_.data()), size * sizeof(int));
}

Text Text::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    int size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));

    std::vector<int32_t> out;
    out.resize(size);
    file.read(reinterpret_cast<char*>(out.data()), size * sizeof(int));

    return Text(std::move(out));
}

int Text::merge(int t1, int t2, int new_token) {
    int merged = 0;
    for (size_t i = 0; i < text_.size(); ++i) {
        if (text_[i] == -1) {
            continue;
        }
        size_t j = skip(i + 1, text_);
        if (j >= text_.size()) {
            break;
        }
        if (text_[i] == t1 && text_[j] == t2) {
            text_[i] = new_token;
            text_[j] = -1;
            merged++;
            dirty_++;
        }
    }
    return merged;
}

TokenPairsCount Text::most_frequent_pair() const {
    TokenPairsCount pairs;
    pairs.reserve(8192);
    size_t i = skip(0, text_);
    size_t j = skip(i + 1, text_);

    while (j < text_.size()) {
        TokenPair this_pair = {text_[i], text_[j]};
        auto& count = pairs[this_pair];
        count++;

        i = j;
        j = skip(i + 1, text_);
    }

    return pairs;
}

void Text::fast_tokenize(const std::vector<TokenPair>& merges,
                         const std::vector<int>* nonterminals) {
    // token -> list of text positions
    std::vector<std::vector<size_t>> token2index;
    // text pos -> token2index idx for that tok
    std::vector<size_t> pos2index;

    auto init = [&]() {
        token2index.resize(merges.size() + 1);
        pos2index.resize(text_.size(), -1);

        // Populate token2index
        for (size_t i = 0; i < text_.size(); ++i) {
            if (text_[i] == -1) {
                continue;
            }
            pos2index[i] = token2index[text_[i]].size();
            token2index[text_[i]].push_back(i);
        }
    };

    init();

    // Merge
    for (size_t token = 0; token < merges.size(); ++token) {
        auto& m = merges[token];
        int a = m.first;
        int b = m.second;

        if (a == -1 || b == -1) {
            continue;
        }

        size_t i = 0;
        while (i < token2index[a].size()) {
            int a_pos = token2index[a][i];
            if (a_pos == -1) {
                ++i;
                continue;
            }

            size_t b_pos = a_pos + 1;
            while (b_pos < text_.size() && text_[b_pos] == -1) {
                ++b_pos;
            }
            if (b_pos >= text_.size() || text_[b_pos] != b) {
                ++i;
                continue;
            }

            text_[a_pos] = token;
            text_[b_pos] = -1;
            token2index[token].push_back(a_pos);
            pos2index[a_pos] = token2index[token].size() - 1;
            token2index[a][i] = -1;
            token2index[b][pos2index[b_pos]] = -1;
            dirty_++;
            ++i;
        }
    }

    // Unmerge nonterminals
    if (nonterminals != nullptr) {
        for (auto it = nonterminals->rbegin(); it != nonterminals->rend();
             ++it) {
            int token = *it;
            int a = merges[token].first;
            int b = merges[token].second;

            if (token == -1) {
                continue;
            }

            for (int pos : token2index[token]) {
                if (pos == -1) continue;

                text_[pos] = a;
                text_[pos + 1] = b;
                pos2index[pos] = token2index[a].size();
                pos2index[pos + 1] = token2index[b].size();
                token2index[a].push_back(pos);
                token2index[b].push_back(pos + 1);
                dirty_--;
            }
        }
    }
}

void Text::unicode_private_to_token() {
    for (size_t i = 0; i < text_.size(); ++i) {
        if (text_[i] == 0xEE || text_[i] == 0xEF) {
            int token = (text_[i] == 0xEE) ? 0xE : 0xF;
            token = (token << 6) + text_[i + 1] - 0x80;
            token = (token << 6) + text_[i + 2] - 0x80;

            if (token >= 0xE000 && token <= 0xF8FF) {
                text_[i] = token - 0xE000;
                text_[i + 1] = -1;
                text_[i + 2] = -1;
                dirty_++;
                dirty_++;
            }
        }
    }
}

void Text::tokenize(const std::vector<TokenPair>& merges) {
    int i = 0;
    for (const auto& tok_pair : merges) {
        int a = tok_pair.first;
        int b = tok_pair.second;
        if (a != -1 && b != -1) {
            merge(a, b, i);
        }

        size_t num_merged = dirty_;
        if (num_merged > text_.size() / 5) {
            std::cout << "GC" << std::endl;
            garbage_collect();
            num_merged = 0;
        }
        ++i;
    }
}
// ------ utils
bool can_merge(const std::string& a, const std::string& b) {
    // Don't merge \n left
    if (a.back() == '\n' && b.front() != '\n') {
        return false;
    }
    // Don't merge digits
    if (isdigit(a.back()) || isdigit(b.front())) {
        return false;
    }
    return true;
}

void sum_counts(TokenPairsCount& dst, const TokenPairsCount& src) {
    for (const auto& p : src) {
        dst[p.first] += p.second;
    }
}

std::string read_full_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << "\n";
        throw std::runtime_error("Error opening file");
    }
    std::string text;

    file.seekg(0, std::ios::end);
    text.reserve(file.tellg());
    file.seekg(0, std::ios::beg);

    text.assign((std::istreambuf_iterator<char>(file)),
                std::istreambuf_iterator<char>());

    return text;
}

const TokenPair* first_valid_pair(TokenPairsCount& counts,
                                  const std::vector<std::string>& vocab) {
    auto pair = counts.end();
    while (true) {
        if (counts.empty()) {
            return nullptr;
        }
        pair = std::max_element(
            counts.begin(), counts.end(), [](const auto& a, const auto& b) {
                return a.second < b.second;
            });

        if (can_merge(vocab[pair->first.first], vocab[pair->first.second])) {
            break;
        }
        counts.erase(pair);
    }
    return &pair->first;
}

TokenPairsCount count_dataset(std::vector<std::string>::const_iterator begin,
                              std::vector<std::string>::const_iterator end,
                              const std::vector<TokenPair>& merges,
                              int threads) {
    if (threads < 2 || std::distance(begin, end) == 1) {
        TokenPairsCount counts;
        for (auto file = begin; file != end; ++file) {
            try {
                std::string text = read_full_file(*file);
                Text t(text);
                // t.unicode_private_to_token();
                t.fast_tokenize(merges);
                sum_counts(counts, t.most_frequent_pair());
            } catch (std::exception& e) {
                std::cerr << "Error reading file: " << *file << "\n";
            }
        }
        return counts;
    } else {
        auto mid = begin + std::distance(begin, end) / 2;
        auto left = std::async(
            std::launch::async, count_dataset, begin, mid, merges, threads / 2);
        auto right = count_dataset(mid, end, merges, threads / 2);
        auto left_counts = left.get();
        sum_counts(left_counts, right);
        return left_counts;
    }
}

std::vector<std::pair<TokenPair, int>> getTop(const TokenPairsCount& counter,
                                              int topN) {
    // Define a min heap
    auto cmp = [](const std::pair<TokenPair, int>& left,
                  const std::pair<TokenPair, int>& right) {
        return left.second > right.second;
    };
    std::priority_queue<std::pair<TokenPair, int>,
                        std::vector<std::pair<TokenPair, int>>,
                        decltype(cmp)>
        minHeap(cmp);

    // Keep only top N elements in the min heap
    for (const auto& kv : counter) {
        minHeap.push(kv);
        if (minHeap.size() > static_cast<size_t>(topN)) {
            minHeap.pop();
        }
    }

    // Extract elements from min heap and add to a vector
    std::vector<std::pair<TokenPair, int>> topTen;
    while (!minHeap.empty()) {
        topTen.push_back(minHeap.top());

        minHeap.pop();
    }

    // Reverse the vector since the elements are in min heap order
    std::reverse(topTen.begin(), topTen.end());
    return topTen;
}

TokenPairsCount speculative_count_dataset(const std::vector<std::string>& files,
                                          const std::vector<TokenPair>& merges,
                                          int threads,
                                          int doc_per_thread,
                                          int topK) {
    TokenPairsCount counts;
    int chunk_size = threads * doc_per_thread;
    for (size_t i = 0; i * chunk_size < files.size(); ++i) {
        auto begin = files.cbegin() + i * chunk_size;
        auto end = files.cbegin() + (i + 1) * chunk_size;
        if (end > files.cend()) {
            end = files.cend();
        }
        auto thread_counts = count_dataset(begin, end, merges, threads);

        if (i == 0) {
            counts = thread_counts;
            continue;
        }

        auto top_before = getTop(counts, topK);
        sum_counts(counts, thread_counts);
        auto top_after = getTop(counts, topK);
        if (std::equal(top_before.begin(),
                       top_before.end(),
                       top_after.begin(),
                       [](const auto& a, const auto& b) {
                           return a.first == b.first;
                       })) {
            std::cout << "Pruning at iteration " << i << "/"
                      << files.size() / chunk_size << " topK=" << topK
                      << std::endl;

            TokenPairsCount pruned;
            for (const auto& p : top_before) {
                pruned[p.first] = p.second;
            }
            return pruned;
        }
    }
    TokenPairsCount pruned;
    for (const auto& p : getTop(counts, topK)) {
        pruned[p.first] = p.second;
    }
    return pruned;
}

std::string binary_to_utf8(const std::string& binary) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    std::basic_string<char32_t> codepoints(
        reinterpret_cast<const uint8_t*>(binary.data()),
        reinterpret_cast<const uint8_t*>(binary.data()) + binary.size());
    assert(codepoints.size() == binary.size());
    return converter.to_bytes(codepoints);
}

std::vector<std::string> vocab_to_utf8(const std::vector<std::string>& vocab) {
    std::vector<std::string> utf8_vocab;
    utf8_vocab.reserve(vocab.size());
    for (const auto& token : vocab) {
        utf8_vocab.push_back(binary_to_utf8(token));
    }
    return utf8_vocab;
}

std::string utf8_to_binary(const std::string& utf8) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    std::basic_string<char32_t> codepoints = converter.from_bytes(utf8);
    std::string binary;
    binary.resize(codepoints.size());
    std::copy(codepoints.begin(), codepoints.end(), binary.begin());
    return binary;
}

std::vector<std::string> utf8_to_vocab(const std::vector<std::string>& utf8) {
    std::vector<std::string> vocab;
    vocab.reserve(utf8.size());
    for (const auto& token : utf8) {
        vocab.push_back(utf8_to_binary(token));
    }
    return vocab;
}

std::pair<std::vector<std::string>, std::vector<TokenPair>> load_vocab(
    const std::string& path) {
    json dump;
    {
        std::ifstream file(path);
        file >> dump;
    }
    std::vector<std::string> vocab;
    std::transform(dump["vocab"].begin(),
                   dump["vocab"].end(),
                   std::back_inserter(vocab),
                   [](const auto& s) {
                       return utf8_to_binary(s.template get<std::string>());
                   });
    std::vector<TokenPair> merges = dump["merges"];
    return {vocab, merges};
}

std::pair<std::vector<std::string>, std::vector<TokenPair>> new_vocab() {
    std::vector<std::string> vocab;
    for (int i = 0; i < 256; ++i) {
        vocab.push_back(std::string(1, i));
    }

    std::vector<TokenPair> merges;
    for (int i = 0; i < 256; ++i) {
        merges.push_back({-1, -1});
    }
    return {vocab, merges};
}
#if 1
int main(int argc, char** argv) {
    // parse args
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset> <vocab.json> <num_new_tokens>" << std::endl;
        return 1;
    }
    int new_num_tokens = std::stoi(argv[3]);

    std::vector<std::string> vocab;
    std::vector<TokenPair> merges;

    if (std::filesystem::exists(argv[2])) {
        std::tie(vocab, merges) = load_vocab(argv[2]);
    } else {
        std::tie(vocab, merges) = new_vocab();
    }

    std::vector<std::string> files;
    for (auto& file : std::filesystem::recursive_directory_iterator(argv[1])) {
        if (file.is_regular_file()) {
            files.push_back(file.path());
        }
    }

    std::random_shuffle(files.begin(), files.end());
    // learn tokens
    int goal_top_k = 1;
    int topK = goal_top_k;
    while (vocab.size() < static_cast<size_t>(new_num_tokens)) {
        TokenPairsCount counts =
            speculative_count_dataset(files, merges, 32 + 16, 6, topK);
        int j = 0;
        for (; j < goal_top_k && !counts.empty(); ++j) {
            auto pair_ptr = first_valid_pair(counts, vocab);
            if (pair_ptr == nullptr) {
                break;
            }
            auto& pair = *pair_ptr;
            merges.push_back(pair);
            vocab.push_back(vocab[pair.first] + vocab[pair.second]);
            std::cout << vocab.size() << ": [" << vocab[pair.first]
                      << "::" << vocab[pair.second]
                      << "] count=" << counts[pair] << std::endl;
            counts.erase(pair);
            std::erase_if(counts, [&](const auto& p) {
                return p.first.first == pair.second ||
                       p.first.second == pair.first;
            });
        }
        topK += (goal_top_k - j) - counts.size();
    }

    // print vocab
    for (size_t i = 256; i < vocab.size(); ++i) {
        std::cout << i << ": " << vocab[i] << std::endl;
    }
    json dump;
    dump["vocab"] = vocab_to_utf8(vocab);
    dump["merges"] = merges;

    std::ofstream file(argv[2]);
    file << dump.dump(4) << std::endl;

    return 0;
}
#endif

