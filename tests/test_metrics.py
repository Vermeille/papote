import math

from papote.metrics import get_dice_rolls, perplexity


class DummyEntropy:
    def __init__(self, value: float) -> None:
        self.value = value

    def exp(self) -> float:
        return math.exp(self.value)


def test_get_dice_rolls():
    text = "10 (1d6) 5 (2d8+1)"
    assert get_dice_rolls(text) == ["", "+1"]


def test_perplexity():
    ce = DummyEntropy(2.0)
    assert math.isclose(perplexity(ce), math.exp(2.0))
