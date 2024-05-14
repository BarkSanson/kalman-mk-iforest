import numpy as np
from sklearn.ensemble import IsolationForest

from online_outlier_detection.base.batch_detector_pipeline import BatchDetectorPipeline
from online_outlier_detection.base.kalman_based_detector_pipeline import KalmanBasedDetectorPipeline
from online_outlier_detection.drift import MannKendallWilcoxonDriftDetector
from online_outlier_detection.window.batch_window import BatchWindow


class MKWKIForestBatchPipeline(BatchDetectorPipeline, KalmanBasedDetectorPipeline):
    def __init__(self,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int):
        super().__init__(score_threshold, alpha, slope_threshold, window_size)
        self.model = IsolationForest()
        self.drift_detector = MannKendallWilcoxonDriftDetector(alpha, slope_threshold)
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

        if self.drift_detector.detect_drift(self.filtered_window.get(), self.filtered_reference_window):
            self._retrain()

            scores = np.abs(self.model.score_samples(self.reference_window.reshape(-1, 1)))
            labels = np.where(scores > self.score_threshold, 1, 0)

            self.window.clear()
            self.filtered_window.clear()
            return scores, labels

        scores = np.abs(self.model.score_samples(self.window.get().reshape(-1, 1)))
        labels = np.where(scores > self.score_threshold, 1, 0)

        self.window.clear()
        self.filtered_window.clear()
        return scores, labels
