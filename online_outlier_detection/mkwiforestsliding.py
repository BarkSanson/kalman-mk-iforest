import numpy as np
from sklearn.ensemble import IsolationForest

from online_outlier_detection.base.sliding_detector import SlidingDetector
from online_outlier_detection.drift.mann_kendall_wilcoxon_drift_detector import MannKendallWilcoxonDriftDetector


class MKWIForestSliding(SlidingDetector):
    def __init__(self,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int):
        super().__init__(score_threshold, alpha, slope_threshold, window_size)

        self.model = IsolationForest()
        self.drift_detector = MannKendallWilcoxonDriftDetector(alpha, slope_threshold)

    def update(self, x) -> tuple[np.ndarray, np.ndarray] | None:
        self.window.append(x)

        if not self.window.is_full():
            return None

        if not self.warm:
            return self._first_training()

        if self.drift_detector.detect_drift(self.window.get(), self.reference_window):
            self._retrain()

        score = np.abs(self.model.score_samples(self.window.get()[-1].reshape(1, -1)))
        label = np.where(score > self.score_threshold, 1, 0)

        return score, label
