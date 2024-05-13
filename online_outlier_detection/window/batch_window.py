import numpy as np

from .window import Window


class BatchWindow(Window):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data = np.array([])

    def append(self, x):
        self.data = np.append(self.data, x)

    def is_full(self):
        return len(self.data) == self.max_size

    def get(self):
        return self.data

    def clear(self):
        self.data = np.array([])
