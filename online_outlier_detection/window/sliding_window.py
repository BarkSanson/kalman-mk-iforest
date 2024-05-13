import numpy as np

from .window import Window


class SlidingWindow(Window):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data = np.array([])

    def append(self, x):
        self.data = np.append(self.data, x)
        if len(self.data) > self.max_size:
            self.data = self.data[1:]

    def is_full(self):
        return len(self.data) == self.max_size

    def get(self):
        return self.data
