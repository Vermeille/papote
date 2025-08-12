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

## Important files and classes
- `papote/model.py` – transformer architecture with `Transformer` (full model),
  `TransformerBlock`, and attention utilities like `SelfAttention` and
  `Rotary`.
- `papote/tokenizer/tokenizer.pyx` – BPE tokenizer core defining `Merges`,
  `Vocab`, and `TextData` classes. Rebuild via `make` after edits.
- `papote/sampler.py` – text generation helpers such as `Sampler`, `TopK`,
  `TopP`, `Temperature`, and `ForbiddenTokens`.
- `papote/train.py` – training utilities including `MLMObjective`,
  `RandomPad`, and `LogCtxLoss` for masked-language modeling and logging.
- `papote/chat.py` and `papote/interactive.py` – command line interfaces for
  chatting or running interactive experiments.
- `model_specs/` – example model definitions like `tiny.py` or `chinchilla.py`
  used to instantiate specific transformer configurations.

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

