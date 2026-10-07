import subprocess
import sys
import os
import json
import sqlite3
from pathlib import Path

import pytest

BPE_PATH = Path("tests/bpe.json").resolve()
DATA_DIR = Path("tests/data").resolve()


@pytest.mark.parametrize("name", ["base", "think", "fim"])
def test_experiment_smoke(name, tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "sample.txt").write_text("hello world")
    cmd = [
        sys.executable,
        "-m",
        "papote.train",
        "--bpe",
        str(BPE_PATH),
        "--data",
        str(DATA_DIR),
        "--model",
        "tiny-1M",
        "--batch-size",
        "2",
        "--global-batch-size",
        "32",
        "--chinchilla-factor",
        "0.001",
        "--experiment",
        name,
        "--max-steps",
        "1",
        "--ctx",
        "16",
        "--test-dir",
        str(test_dir),
        "--trackio-name",
        name,
    ]
    trackio_dir = tmp_path / "trackio"
    env = {
        **os.environ,
        "TRACKIO_DIR": str(trackio_dir),
        "TORCHDYNAMO_DISABLE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    # The integration test uses local storage even when the caller tracks remotely.
    env.pop("TRACKIO_SPACE_ID", None)
    env.pop("TRACKIO_SERVER_URL", None)
    result = subprocess.run(
        cmd, cwd=tmp_path, env=env, timeout=120, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with sqlite3.connect(trackio_dir / "papote.db") as db:
        rows = db.execute("SELECT metrics FROM metrics").fetchall()
        runs = db.execute("SELECT run_name, config FROM configs").fetchall()
    metrics = [json.loads(row[0]) for row in rows]
    assert any("train/loss" in row for row in metrics)
    assert any(row.get("train/num_tokens", 0) > 0 for row in metrics)
    assert any("train/loss_at_pos" in row for row in metrics)
    assert len(runs) == 1
    assert runs[0][0] == name
    assert json.loads(runs[0][1])["experiment"] == name
