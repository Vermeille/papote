# Papote

[![CI](https://github.com/.../papote/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/.../papote/actions/workflows/ci.yml)

Papote is a full reimplementation from scratch of generative language models.
It is mainly done as way to learn in depth the intricacies of writing and
training language models. The end goal is to train it on some conversation
archives and have a virtual double of myself.

It includes:

- A BPE reimplementation that may or may not be 100% correct, written in Cython
  for efficienty. Training and inference code.
- A base transformer with various architecture configurations.
- An interactive interface to play with prompting. And a chat interface.

Anyway, don't use Papote. It's not meant to be used in production. I'm just
building skill and understanding. Use HuggingFace.

## Installation

Papote can be installed from source using standard Python packaging tools.

Using `pip`:

```bash
pip install .
```

Using [`uv`](https://github.com/astral-sh/uv):

```bash
uv pip install .
```

## Development with `uv`

Papote manages its dependencies with [`uv`](https://github.com/astral-sh/uv).
After cloning the repository, install the project requirements with:

```bash
uv sync
```

This creates a `.venv` directory containing all dependencies. Run commands inside
this environment using `uv run`, for example:

```bash
uv run pytest                 # run the test suite
uv run python papote/chat.py  # start the chat interface
```
