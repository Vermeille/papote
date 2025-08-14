import random


class RandomCase:
    def __init__(self, lower_token, upper_token, lowercase_p=0.25, uppercase_p=0.25):
        self.lowercase_p = lowercase_p
        self.uppercase_p = uppercase_p
        self.lower_token = lower_token
        self.upper_token = upper_token

    def __call__(self, text):
        p = random.uniform(0, 1)
        if p < self.lowercase_p:
            text = self.lower_token + text.lower()
        elif p < self.lowercase_p + self.uppercase_p:
            text = self.upper_token + text.upper()
        return text
