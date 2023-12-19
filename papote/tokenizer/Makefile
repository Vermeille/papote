SOURCES=main.cpp

all: main cython

main: text.h token_pair.h json.hpp $(SOURCES)
	g++ -O3 -std=c++20 -march=native -o main $(SOURCES)


cython: tokenizer.cpython-310-x86_64-linux-gnu.so
tokenizer.cpython-310-x86_64-linux-gnu.so: tokenizer.pyx
	cythonize -bi $<

