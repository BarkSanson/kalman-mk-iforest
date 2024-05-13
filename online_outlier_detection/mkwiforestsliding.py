import numpy as np
from pymannkendall import yue_wang_modification_test
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest

from online_outlier_detection.sliding_detector import SlidingDetector


class MKWIForestSliding(SlidingDetector):
    def __init__(self,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int):
        super().__init__(score_threshold, alpha, slope_threshold, window_size)
        self.model = IsolationForest()

    def update(self, x) -> tuple[np.ndarray, np.ndarray] | None:
        self.window.append(x)

        if not self.window.is_full():
            return None

        if not self.warm:
            return self._first_training()

        _, h, _, _, _, _, _, slope, _ = \
            yue_wang_modification_test(self.window.get())
        d = np.around(self.window.get() - self.reference_window, decimals=3)
        stat, p_value = wilcoxon(d)

        # Data distribution is changing enough to retrain the model
        if (h and abs(slope) >= self.slope_threshold) or p_value < self.alpha:
            self._retrain()

        score = np.abs(self.model.score_samples(self.window.get()[-1].reshape(1, -1)))
        label = np.where(score > self.score_threshold, 1, 0)

        return score, label

    def _retrain(self):
        self.reference_window = self.window.get().copy()
        self.model.fit(self.reference_window.reshape(-1, 1))
        self.retrains += 1
        print(f"Retraining model... Number of retrains: {self.retrains}")
