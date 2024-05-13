import numpy as np
from pymannkendall import yue_wang_modification_test
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest

from online_outlier_detection.kalman_based_detector import KalmanBasedDetector
from online_outlier_detection.sliding_detector import SlidingDetector
from online_outlier_detection.window.sliding_window import SlidingWindow


class MKWKIForestSliding(SlidingDetector, KalmanBasedDetector):
    def __init__(self,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int):
        super().__init__(score_threshold, alpha, slope_threshold, window_size)
        self.model = IsolationForest()

        self.filtered_window = SlidingWindow(self.window_size)

    def update(self, x) -> tuple[np.ndarray, np.ndarray] | None:
        self.window.append(x)

        # Apply Kalman filter to current data
        self.kf.predict()
        self.kf.update(x)

        filtered_x = self.kf.x

        self.filtered_window.append(filtered_x)

        if not self.window.is_full():
            return None

        if not self.warm:
            self.filtered_reference_window = self.filtered_window.get().copy()
            scores, labels = self._first_training()

            return scores, labels

        _, h, _, _, _, _, _, slope, _ = \
            yue_wang_modification_test(self.filtered_window.get())
        d = np.around(self.filtered_window.get() - self.filtered_reference_window, decimals=3)
        stat, p_value = wilcoxon(d)

        # Data distribution is changing enough to retrain the model
        if (h and abs(slope) >= self.slope_threshold) or p_value < self.alpha:
            self._retrain()

        score = np.abs(self.model.score_samples(self.window.get()[-1].reshape(1, -1)))
        label = np.where(score > self.score_threshold, 1, 0)

        return score, label
