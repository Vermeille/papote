import pytest
from typing import Optional

from papote.utils import OptionsBase, txt_extensions


class MyOptions(OptionsBase):
    a: int = 1
    b: Optional[int] = None


def test_options_base_parse_sets_attribute():
    opts = MyOptions()
    opts.parse("a 5")
    assert opts.a == 5


def test_options_base_parse_union_none():
    opts = MyOptions()
    opts.parse("b 7")
    assert opts.b == 7
    opts.parse("b")
    assert opts.b is None


def test_options_base_parse_unknown_option():
    opts = MyOptions()
    with pytest.raises(ValueError):
        opts.parse("c 3")


def test_txt_extensions_contains_common():
    assert ".txt" in txt_extensions
    assert ".md" in txt_extensions
