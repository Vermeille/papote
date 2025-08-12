# distutils: language = c++
# distutils: sources = tokenizer/main.cpp
# distutils: extra_compile_args = -std=c++20
# cython: language_level = 3

from tokenizer cimport TokenPair, Text
from libcpp.vector cimport vector
from libcpp.string cimport string

from typing import Optional

cdef TokenPair make_token_pair(int first, int second):
    return TokenPair(first, second)

cdef (int, int) read_token_pair(TokenPair token_pair):
    return token_pair.first, token_pair.second

cdef class Merges:
    cdef vector[TokenPair] merges

    def __init__(self):
        self.merges = vector[TokenPair]()

    def add(self, int first, int second):
        self.merges.push_back(make_token_pair(first, second))

    def __iter__(self):
        cdef int i
        for i in range(self.merges.size()):
            yield read_token_pair(self.merges[i])

    def __len__(self):
        return self.merges.size()

    def to_list(self):
        cdef int i
        cdef list result = []
        for i in range(self.merges.size()):
            result.append(read_token_pair(self.merges[i]))
        return result

    @staticmethod
    def from_list(list merges):
        cdef Merges result = Merges()
        for first, second in merges:
            result.add(first, second)
        return result

cdef class Vocab:
    cdef vector[string] vocab

    def __init__(self):
        self.vocab = vector[string]()

    @staticmethod
    def from_list(list vocab):
        cdef Vocab result = Vocab()
        for word in vocab:
            result.vocab.push_back(word)
        return result

    def __iter__(self):
        cdef int i
        for i in range(self.vocab.size()):
            yield self.vocab[i]

    def __len__(self):
        return self.vocab.size()

    def to_list(self):
        cdef int i
        cdef list result = []
        for i in range(self.vocab.size()):
            result.append(self.vocab[i])
        return result


cdef class TextData:
    cdef Text text

    def __init__(self, str text):
        self.text = Text(text.encode('utf-8'))

    def __len__(self):
        return self.text.size()

    def as_str_tokens(self, Vocab vocab):
        return self.text.as_str_tokens(vocab.vocab)

    def as_str(self, Vocab vocab, str sep=None):
        cdef string sep_str
        if sep is None:
            return self.text.as_str(vocab.vocab, NULL)
        else:
            sep_str = sep.encode('utf-8')
            return self.text.as_str(vocab.vocab, &sep_str)

    def unicode_private_to_token(self, Vocab vocab):
        return self.text.unicode_private_to_token()

    def fast_tokenize(self, Merges merges, Vocab non_terminals):
        if non_terminals is None:
            return self.text.fast_tokenize(merges.merges, NULL)
        else:
            assert False, "non_terminals is not None, not supported yet"

