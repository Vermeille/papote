import sys
from pathlib import Path

import pytest
from papote.model import Rotary, RotarySingle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def clear_rotary_cache():
    Rotary._cache.clear()
    RotarySingle._cache.clear()
