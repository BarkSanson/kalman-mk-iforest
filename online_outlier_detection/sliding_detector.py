import numpy as np

from online_outlier_detection.base_detector import BaseDetector
from online_outlier_detection.window.sliding_window import SlidingWindow


class SlidingDetector(BaseDetector):
    def __init__(self,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int):
        super().__init__(score_threshold, alpha, slope_threshold, window_size)

        self.window = SlidingWindow(self.window_size)

    def update(self, x) -> tuple[np.ndarray, np.ndarray] | None:
        pass

    def _check_retrain_and_predict(self, h, slope, p_value) -> tuple[np.ndarray, np.ndarray]:
        if (h and abs(slope) >= self.slope_threshold) or p_value < self.alpha:
            self._retrain()

        score = np.abs(self.model.score_samples(self.window.get()[-1].reshape(1, -1)))
        label = np.where(score > self.score_threshold, 1, 0)

        return score, label
