from abc import ABC, abstractmethod

import numpy as np


class BaseDriftDetector(ABC):
    @abstractmethod
    def detect_drift(self, x: np.ndarray, y: np.ndarray) -> bool:
        pass
