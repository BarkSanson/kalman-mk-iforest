import numpy as np
from filterpy.kalman import KalmanFilter
from pymannkendall import yue_wang_modification_test
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest


class MKWKIForestBatch:
    def __init__(self,
                 score_threshold: float = 0.75,
                 alpha: float = 0.05,
                 slope_threshold: float = 0.001,
                 window_size: int = 64):
        self.model = IsolationForest()
        self.kf = KalmanFilter(dim_x=1, dim_z=1)
        self.kf.Q = 0.001
        self.kf.F = np.array([[1]])
        self.kf.H = np.array([[1]])
        self.kf.x = np.array([0])
        self.kf.P = np.array([1])

        self.alpha = alpha
        self.slope_threshold = slope_threshold
        self.window_size = window_size
        self.score_threshold = score_threshold

        self.raw_window = np.array([])
        self.filtered_window = np.array([])

        self.reference_window = np.array([])
        self.filtered_reference_window = np.array([])

        self.warm = False

        self.retrains = 0

    def update(self, x) -> np.ndarray | None:
        self.raw_window = np.append(self.raw_window, x)

        # Apply Kalman filter to current data
        self.kf.predict()
        self.kf.update(x)

        filtered_x = self.kf.x

        self.filtered_window = np.append(self.filtered_window, filtered_x)

        if len(self.raw_window) < self.window_size:
            return None

        if not self.warm:
            self.reference_window = self.raw_window.copy()
            self.filtered_reference_window = self.filtered_window.copy()

            ref = self.reference_window.reshape(-1, 1)
            self.model.fit(ref)

            scores = np.abs(self.model.score_samples(ref))
            labels = np.where(scores > self.score_threshold, 1, 0)

            self.warm = True
            self.raw_window = np.array([])
            self.filtered_window = np.array([])

            return labels

        _, h, _, _, _, _, _, slope, _ = \
            yue_wang_modification_test(self.filtered_window)
        d = np.around(self.raw_window - self.reference_window, decimals=3)
        stat, p_value = wilcoxon(d)

        # If the water level is rising or decreasing significantly, or the data is significantly different from the
        # reference, retrain the model
        if (h and abs(slope) >= self.slope_threshold) or p_value < self.alpha:
            self._retrain()

            scores = self.model.score_samples(self.reference_window.reshape(-1, 1))
            labels = np.where(scores > self.score_threshold, 1, 0)

            self.raw_window = np.array([])
            self.filtered_window = np.array([])
            return labels

        scores = self.model.score_samples(self.raw_window.reshape(-1, 1))
        labels = np.where(scores > self.score_threshold, 1, 0)

        self.raw_window = np.array([])
        self.filtered_window = np.array([])
        return labels

    def _retrain(self):
        self.reference_window = self.raw_window.copy()
        self.filtered_reference_window = self.filtered_window.copy()
        self.model.fit(self.reference_window.reshape(-1, 1))
        self.retrains += 1
        print(f"Retraining model... Number of retrains: {self.retrains}")
