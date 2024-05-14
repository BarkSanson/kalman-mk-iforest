from abc import ABC, abstractmethod

import numpy as np

from online_outlier_detection.window import Window
from online_outlier_detection.drift import BaseDriftDetector


class BaseDetectorPipeline(ABC):
    def __init__(self,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int):
        self.model = None

        self.alpha = alpha
        self.slope_threshold = slope_threshold
        self.window_size = window_size
        self.score_threshold = score_threshold

        self.window: Window = None # Dunno if this is the most 'pythonic' way to do this
        self.drift_detector: BaseDriftDetector = None # Same
        self.reference_window = np.array([])

        self.warm = False

        self.retrains = 0

    @abstractmethod
    def update(self, x) -> tuple[np.ndarray, np.ndarray] | None:
        pass

    def _first_training(self):
        self.reference_window = self.window.get().copy()

        ref = self.reference_window.reshape(-1, 1)
        self.model.fit(ref)

        scores = np.abs(self.model.score_samples(ref))
        labels = np.where(scores > self.score_threshold, 1, 0)

        self.warm = True

        return scores, labels

    def _retrain(self):
        self.reference_window = self.window.get().copy()
        self.model.fit(self.reference_window.reshape(-1, 1))
        self.retrains += 1
        print(f"Retraining model... Number of retrains: {self.retrains}")
