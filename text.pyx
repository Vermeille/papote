#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
from libc.stdlib cimport rand, RAND_MAX
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from cython.operator cimport dereference as deref
from typing import List

cdef float random():
    return float(rand()) / RAND_MAX

cdef class Text:
    cdef vector[int] text

    def __init__(self, text):
        self.text = text.encode('utf-8')

    @staticmethod
    def from_bytes(text: bytes):
        t = Text('')
        t.text = text
        return t

    def _as_tokens(self):
        return self.text

    def set_tokens(self, tokens: List[int]):
        self.text = tokens

    def as_tokens(self):
        self.garbage_collect()
        return self._as_tokens()

    def garbage_collect(self):
        cdef int i
        cdef int j
        cdef int size = self.text.size()
        cdef vector[int] new_text
        new_text.reserve(size)
        for i in range(size):
            if self.text[i] != -1:
                new_text.push_back(self.text[i])
        self.text = new_text

    def merge(self, int t1, int t2, int new_token, float dropout=0.0) -> int:
        cdef int i
        cdef int j
        cdef int merged = 0
        while i < self.text.size():
            while i < self.text.size() and self.text[i] != t1:
                i += 1
            j = i + 1
            while j < self.text.size() and self.text[j] == -1:
                j += 1
            if j < self.text.size() and self.text[j] == t2:
                if not random() < dropout:
                    self.text[i] = new_token
                    self.text[j] = -1
                    merged += 1
            i = j
        return merged

    def as_bytes_tokens(self, vocab: list[bytes]):
        cdef int i
        self.garbage_collect()
        out = []
        for i in range(self.text.size()):
            o = self.text[i]
            o2 = vocab[o]
            out.append(o2)
        return out

    def as_str_tokens(self, vocab: list[bytes]):
        cdef int i
        self.garbage_collect()
        out = []
        for i in range(self.text.size()):
            o = self.text[i]
            o2 = vocab[o].decode('utf-8', errors='ignore')
            out.append(o2)
        return out

    def as_str(self, vocab: list[bytes]):
        return b''.join(self.as_str_tokens(vocab)).decode('utf-8')

    @staticmethod
    cdef inline int skip(int i, int* t, int size):
        while i < size and t[i] == -1:
            i += 1
        return i

    def most_frequent_pair(self):
        cdef int i
        cdef int j
        cdef int size = self.text.size()
        cdef map[pair[int, int], int] pairs
        cdef pair[int, int] this_pair
        cdef int count
        cdef int max_count = 0
        cdef map[pair[int, int], int].iterator it
        cdef pair[int, int] max_pair = pair[int, int](0, 0)
        cdef int* t = self.text.data()
        i = Text.skip(0, t, size)
        j = Text.skip(i + 1, t, size)
        while j < size:
            this_pair.first = t[i]
            this_pair.second = t[j]
            it = pairs.find(this_pair)
            if it == pairs.end():
                count = 1
                pairs[this_pair] = count
            else:
                count = deref(it).second + 1
                deref(it).second = count

            if count > max_count:
                max_count = count
                max_pair = this_pair
            i = j
            j = Text.skip(i + 1, t, size)

        return max_pair, pairs

    def unicode_private_to_token(self):
        # https://en.wikipedia.org/wiki/Private_Use_Areas
        # Find bytes of the form \xee\x80\x80 corresponding to code points of
        # the form U+Exxx and replace them with a single token with value xxx

        cdef int i
        cdef int j
        cdef int size = self.text.size()
        cdef int* t = self.text.data()
        cdef int token = 0
        cdef int count = 0
        for i in range(size):
            if t[i] == 0xEE or t[i] == 0xEF:
                token = 0xE if t[i] == 0xEE else 0xF
                token <<= 6
                token += t[i + 1] - 0x80
                token <<= 6
                token += t[i + 2] - 0x80
                if token >= 0xE000 and token <= 0xF8FF:
                    t[i] = token - 0xE000
                    t[i + 1] = -1
                    t[i + 2] = -1

    def fast_tokenize(self, merges, float dropout=0.0):
        cdef vector[vector[int]] token2index
        cdef vector[int] pos2index

        # populate token2index
        cdef int i
        cdef int j
        cdef int size = self.text.size()
        cdef int* t = self.text.data()
        cdef int token
        cdef int a
        cdef int a_pos
        cdef int b
        cdef int b_pos
        cdef int begin
        cdef int end

        token2index.resize(len(merges) + 1)
        pos2index.resize(size)
        for i in range(size):
            token = t[i]
            if token != -1:
                pos2index[i] = token2index[token].size()
                token2index[token].push_back(i)

        # merge
        for token, (a, b) in enumerate(merges):
            if a == -1 or b == -1:
                continue

            i = 0
            while i < token2index[a].size():
                a_pos = token2index[a][i]
                if a_pos == -1:
                    i += 1
                    continue
                b_pos = a_pos + 1

                while b_pos < size and t[b_pos] == -1:
                    b_pos += 1
                if b_pos >= size or t[b_pos] != b:
                    i += 1
                    continue

                if not random() < dropout:
                    # merge
                    t[a_pos] = token
                    t[b_pos] = -1
                    # update token2index
                    token2index[token].push_back(a_pos)
                    if False:
                        token2index[a].erase(token2index[a].begin() + i)
                        for j in range(token2index[b].size()):
                            if token2index[b][j] == b_pos:
                                token2index[b].erase(token2index[b].begin() + j)
                                break
                    else:
                        pos2index[a_pos] = token2index[token].size() - 1
                        token2index[a][i] = -1
                        token2index[b][pos2index[b_pos]] = -1
                    i -= 1
                i += 1


    def tokenize(self, merges, float dropout=0.0):
        for i, (a, b) in enumerate(merges):
            if a == -1 or b == -1:
                continue
            self.merge(a, b, i, dropout)

