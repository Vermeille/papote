import re


def get_dice_rolls(txt):
    roll = re.compile(r'\d+ \(\d+d\d+(\+\d)?\)')
    return roll.findall(txt)


def perplexity(cross_entropy):
    return cross_entropy.exp()
