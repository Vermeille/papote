# Papote Codebase Guide

This repository contains a from‑scratch implementation of language-model tooling
and experiments. The `papote` package houses most of the code; `model_specs`
contains Python modules describing various transformer configurations.

## Navigation
- Use `rg` (ripgrep) to search through the source tree.
- Core modules live under `papote/` (tokenizer, models, training scripts,
  evaluation utilities).
- `papote/tokenizer/` includes a C++/Cython BPE implementation. When modifying
  files in this folder, run `make` to rebuild the extension.

## Running tests
Basic smoke tests live in `papote/test_all.py`. Run them with:

```bash
python -m papote.test_all
```

This script exercises tokenization and sampling helpers. Some dependencies such
as `tqdm`, `torch`, and `torchelie` must be available. If a dependency is
missing, install it or note the failure in your report.

## Usage tips
- Interactive experiments can be launched with `python papote/interactive.py` or
  the chat interface `python papote/chat.py`.
- Model specification examples are in `model_specs/` and can be imported to
  create transformers for training or evaluation.
- When updating documentation or code, keep the style consistent with existing
  files (PEP8 for Python).

