from abc import ABC, abstractmethod


class Window(ABC):
    def __init__(self):
        self.data = None

    @abstractmethod
    def append(self, x):
        pass

    @abstractmethod
    def is_full(self):
        pass

    @abstractmethod
    def get(self):
        pass

    def __len__(self):
        return len(self.data)