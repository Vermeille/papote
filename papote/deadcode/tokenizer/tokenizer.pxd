from libc.stdint cimport int32_t
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

cdef extern from "token_pair.h" nogil:
    ctypedef unordered_map[string, int] TokenPairsCount
    cdef cppclass TokenPair:
        TokenPair()
        TokenPair(int, int)
        TokenPair(pair[int, int])

        int first
        int second

cdef extern from "text.h" nogil:
    cdef cppclass Text:
        Text()
        Text(const string& text)
        int32_t size()
        vector[string] as_str_tokens(const vector[string]& vocab)
        string as_str(const vector[string]& vocab, string* sep)
        TokenPairsCount most_frequent_pair()
        void unicode_private_to_token()
        void tokenize(const vector[TokenPair]& merges)
        void fast_tokenize(const vector[TokenPair]& merges,
                const vector[int]* non_terminals)

