import numpy as np
from pymannkendall import yue_wang_modification_test
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest


class MKWIForestSliding:
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
        if len(self.raw_window) < self.window_size:
            self.raw_window = np.append(self.raw_window, x)
            return None
        else:
            self.raw_window = np.roll(self.raw_window, -1)
            self.raw_window[-1] = x

        if not self.warm:
            self.reference_window = self.raw_window.copy()

            ref = self.reference_window.reshape(-1, 1)
            self.model.fit(ref)

            scores = np.abs(self.model.score_samples(ref))
            labels = np.where(scores > self.score_threshold, 1, 0)

            self.warm = True

            return scores, labels

        _, h, _, _, _, _, _, slope, _ = \
            yue_wang_modification_test(self.raw_window)
        d = np.around(self.raw_window - self.reference_window, decimals=3)
        stat, p_value = wilcoxon(d)

        # If the water level is rising or decreasing significantly, or the data is significantly different from the
        # reference, retrain the model
        if (h and abs(slope) >= self.slope_threshold) or p_value < self.alpha:
            self._retrain()

        score = np.abs(self.model.score_samples(self.raw_window[-1].reshape(1, -1)))
        label = np.where(score > self.score_threshold, 1, 0)

        return score, label

    def _retrain(self):
        self.reference_window = self.raw_window.copy()
        self.model.fit(self.raw_window.reshape(-1, 1))
        self.retrains += 1
        print(f"Retraining model... Number of retrains: {self.retrains}")
