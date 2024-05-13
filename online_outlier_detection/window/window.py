from abc import ABC, abstractmethod


class Window(ABC):
    @abstractmethod
    def append(self, x):
        pass

    @abstractmethod
    def is_full(self):
        pass

    @abstractmethod
    def get(self):
        pass
