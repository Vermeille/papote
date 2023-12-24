#include <functional>
#include <tuple>
#include <unordered_map>

inline int hash_combine(int lhs, int rhs) {
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
}

namespace std {
//----------- TokenPair
template <>
struct hash<std::pair<int, int>> {
    int operator()(const std::pair<int32_t, int32_t>& p) const {
        return hash_combine(std::hash<int32_t>()(p.first),
                            std::hash<int32_t>()(p.second));
    }
};
}  // namespace std
