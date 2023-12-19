#pragma once

#include <utility>

struct TokenPair {
    TokenPair() = default;
    TokenPair(const std::pair<int32_t, int32_t>& p)
        : first(p.first), second(p.second) {}
    TokenPair(int32_t first, int32_t second) : first(first), second(second) {}
    int32_t first;
    int32_t second;
    bool operator==(const TokenPair& rhs) const = default;
    TokenPair& operator=(const TokenPair& rhs) = default;
};

namespace std {
template <>
struct hash<TokenPair> {
    int operator()(const TokenPair& p) const;
};
}  // namespace std

