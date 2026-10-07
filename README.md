# Papote
[![Tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

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

## Experiment tracking

Training logs to [Trackio](https://github.com/gradio-app/trackio) locally by
default, under the `papote` project. No tracking server or account is required.
Only rank zero creates a run during distributed training; CPU training is also
tracked. Each run records the training configuration and flushes its logs when
training finishes or raises an exception.

Use `--trackio-project NAME` and `--trackio-name NAME` with `python -m papote.train`
to select a project and name a run. Omit the run name to generate a unique one,
or pass `--no-trackio` to disable tracking.

```bash
uv run trackio show --project papote
```

Metrics use `train/` and `test/` prefixes and the training iteration as their
step. Training metrics are logged every 10 batches and after the final batch
of each epoch; evaluation metrics are logged after evaluation. Position losses
and weights are stored as tables with `position` and `value` columns. Generated
text and best/worst examples are stored as HTML reports.

Set `TRACKIO_DIR` before training or opening the dashboard to use a custom
storage directory (for example, `/workspace/trackio` in the cloud environment).
Tracking is provided by Torchelie's Trackio callbacks. Papote owns the shared
run so it can record its name and configuration and close it on completion or
failure. `uv sync` installs the published Torchelie commit pinned in
`pyproject.toml`; a sibling checkout is no longer needed. Visdom remains an
optional feature of Torchelie and is not installed by Papote.
