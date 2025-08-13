import subprocess
import sys
import tempfile
from pathlib import Path

BPE_PATH = Path('tests/bpe.json').resolve()
DATA_DIR = Path('tests/data').resolve()


def run_experiment(name: str):
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp) / 'test'
        test_dir.mkdir()
        (test_dir / 'sample.txt').write_text('hello world')
        cmd = [
            sys.executable,
            '-m',
            'papote.train',
            '--bpe', str(BPE_PATH),
            '--data', str(DATA_DIR),
            '--model', 'tiny-1M',
            '--batch-size', '2',
            '--global-batch-size', '32',
            '--chinchilla-factor', '0.001',
            '--experiment', name,
            '--max-steps', '1',
        ]
        subprocess.run(cmd, check=True, cwd=tmp)


def test_experiment_smoke():
    for name in ('base', 'think', 'fim'):
        run_experiment(name)

