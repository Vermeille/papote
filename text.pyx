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

    def merge(self, int t1, int t2, int new_token, float dropout=0.0) -> bool:
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
                    merged = 1
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

    def tokenize(self, merges, float dropout=0.0):
        for i, (a, b) in enumerate(merges):
            self.merge(a, b, i+256, dropout)

