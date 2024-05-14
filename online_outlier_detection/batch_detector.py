import numpy as np

from .base_detector import BaseDetector
from .window.batch_window import BatchWindow


class BatchDetector(BaseDetector):
    def __init__(self,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int):
        super().__init__(score_threshold, alpha, slope_threshold, window_size)

        self.window = BatchWindow(self.window_size)

    def _first_training(self):
        scores, labels = super()._first_training()
        self.window.clear()

        return scores, labels

    def update(self, x) -> tuple[np.ndarray, np.ndarray] | None:
        pass

    def _check_retrain_and_predict(self, h, slope, p_value):
        if (h and abs(slope) >= self.slope_threshold) or p_value < self.alpha:
            self._retrain()

            scores = np.abs(self.model.score_samples(self.reference_window.reshape(-1, 1)))
            labels = np.where(scores > self.score_threshold, 1, 0)

            self.window.clear()
            return scores, labels

        scores = np.abs(self.model.score_samples(self.window.get().reshape(-1, 1)))
        labels = np.where(scores > self.score_threshold, 1, 0)

        self.window.clear()
        return scores, labels

