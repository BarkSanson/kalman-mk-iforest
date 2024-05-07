import numpy as np
from pymannkendall import yue_wang_modification_test
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest


class MKWIForestBatch:
    def __init__(self,
                 score_threshold: float = 0.75,
                 alpha: float = 0.05,
                 slope_threshold: float = 0.001,
                 window_size: int = 64):
        self.model = IsolationForest()

        self.alpha = alpha
        self.slope_threshold = slope_threshold
        self.window_size = window_size
        self.score_threshold = score_threshold

        self.raw_window = np.array([])
        self.reference_window = np.array([])

        self.warm = False

        self.retrains = 0

    def update(self, x) -> tuple[np.ndarray, np.ndarray] | None:
        self.raw_window = np.append(self.raw_window, x)

        if len(self.raw_window) < self.window_size:
            return None

        if not self.warm:
            self.reference_window = self.raw_window.copy()

            ref = self.reference_window.reshape(-1, 1)
            self.model.fit(ref)

            scores = np.abs(self.model.score_samples(ref))
            labels = np.where(scores > self.score_threshold, 1, 0)

            self.warm = True
            self.raw_window = np.array([])

            return scores, labels

        _, h, _, _, _, _, _, slope, _ = \
            yue_wang_modification_test(self.raw_window)
        d = np.around(self.raw_window - self.reference_window, decimals=3)
        stat, p_value = wilcoxon(d)

        # If the water level is rising or decreasing significantly, or the data is significantly different from the
        # reference, retrain the model
        if (h and abs(slope) >= self.slope_threshold) or p_value < self.alpha:
            self._retrain()

            scores = np.abs(self.model.score_samples(self.reference_window.reshape(-1, 1)))
            labels = np.where(scores > self.score_threshold, 1, 0)

            self.raw_window = np.array([])
            return scores, labels

        scores = np.abs(self.model.score_samples(self.raw_window.reshape(-1, 1)))
        labels = np.where(scores > self.score_threshold, 1, 0)

        self.raw_window = np.array([])
        return scores, labels

    def _retrain(self):
        self.reference_window = self.raw_window.copy()
        self.model.fit(self.reference_window.reshape(-1, 1))
        self.retrains += 1
        print(f"Retraining model... Number of retrains: {self.retrains}")
