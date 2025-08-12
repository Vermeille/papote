# Papote

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
