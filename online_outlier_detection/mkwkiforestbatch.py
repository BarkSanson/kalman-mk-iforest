import numpy as np
from filterpy.kalman import KalmanFilter
from pymannkendall import yue_wang_modification_test
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest

from online_outlier_detection.batch_detector import BatchDetector
from online_outlier_detection.window.batch_window import BatchWindow
from online_outlier_detection.kalman_based_detector import KalmanBasedDetector


class MKWKIForestBatch(BatchDetector, KalmanBasedDetector):
    def __init__(self,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int):
        super().__init__(score_threshold, alpha, slope_threshold, window_size)
        self.model = IsolationForest()

        self.filtered_window = BatchWindow(window_size)

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

            self.filtered_window.clear()

            return scores, labels

        _, h, _, _, _, _, _, slope, _ = \
            yue_wang_modification_test(self.filtered_window.get())
        d = np.around(self.filtered_window.get() - self.filtered_reference_window, decimals=3)
        stat, p_value = wilcoxon(d)

        scores, labels = self._check_retrain_and_predict(h, slope, p_value)
        self.filtered_window.clear()

        return scores, labels
