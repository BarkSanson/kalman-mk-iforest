import collections

import numpy as np
from filterpy.kalman import KalmanFilter
from sklearn.ensemble import IsolationForest
from pymannkendall import yue_wang_modification_test
from scipy.stats import wilcoxon


class MKKalmanIForestPipeline:
    def __init__(self,
                 alpha: float = 0.05,
                 contamination: float = "auto",
                 slope_threshold: float = 0.001,
                 window_size: int = 64):
        self.model = IsolationForest(contamination=contamination)
        self.kf = KalmanFilter(dim_x=1, dim_z=1)
        self.kf.Q = 0.001
        self.kf.F = np.array([[1]])
        self.kf.H = np.array([[1]])
        self.kf.x = np.array([0])
        self.kf.P = np.array([1])

        self.alpha = alpha
        self.slope_threshold = slope_threshold
        self.window_size = window_size

        self.raw_sliding_window = collections.deque(maxlen=window_size)
        self.filtered_sliding_window = collections.deque(maxlen=window_size)

        self.reference_window = np.array([])
        self.filtered_reference_window = np.array([])

        self.warm = False

        self.retrains = 0

    def update(self, x):
        self.raw_sliding_window.append(x)

        # Apply Kalman filter to current data
        self.kf.predict()
        self.kf.update(x)

        filtered_x = self.kf.x

        self.filtered_sliding_window.append(filtered_x)

        if len(self.raw_sliding_window) < self.window_size:
            return None

        if not self.warm:
            self.reference_window = np.array(self.raw_sliding_window)
            self.filtered_reference_window = np.array(self.filtered_sliding_window)

            self.model.fit(self.reference_window.reshape(-1, 1))

            scores = self.model.score_samples(self.reference_window.reshape(-1, 1))

            self.warm = True

            return scores

        _, h, _, _, _, _, _, slope, _ = \
            yue_wang_modification_test(self.filtered_sliding_window)
        #stat, p_value = wilcoxon(self.filtered_reference_window, self.filtered_sliding_window)

        # If the water level is rising or decreasing significantly, or the data is significantly different from the
        # reference, retrain the model
        if h and abs(slope) >= self.slope_threshold:# or (p_value < self.alpha and stat > 0.1):
            self._retrain()

        score = self.model.score_samples(x.reshape(1, -1))
        return score

    def _retrain(self):
        self.reference_window = np.array(self.raw_sliding_window)
        self.filtered_reference_window = np.array(self.filtered_sliding_window)
        self.model.fit(self.reference_window.reshape(-1, 1))
        self.retrains += 1
        print(f"Retraining model... Number of retrains: {self.retrains}")
