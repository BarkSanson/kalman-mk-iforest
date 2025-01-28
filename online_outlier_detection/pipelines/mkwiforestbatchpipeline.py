import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Union, Tuple

from online_outlier_detection.pipelines.base.batch_detector_pipeline import BatchDetectorPipeline
from online_outlier_detection.drift import MannKendallWilcoxonDriftDetector


class MKWIForestBatchPipeline(BatchDetectorPipeline):
    def __init__(self,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int):
        super().__init__(score_threshold, alpha, slope_threshold, window_size)
        self.model = IsolationForest()
        self.drift_detector = MannKendallWilcoxonDriftDetector(alpha, slope_threshold)

    def update(self, x) -> Union[Tuple[np.ndarray, bool], int]:
        self.window.append(x)

        if not self.window.is_full():
            return len(self.window) - 1

        if not self.warm:
            return self._first_training(), False

        if self.drift_detector.detect_drift(self.window.get(), self.reference_window):
            self._retrain()

            scores = np.abs(self.model.score_samples(self.reference_window.reshape(-1, 1)))
            labels = np.where(scores > self.score_threshold, 1, 0)

            result = np.c_[self.reference_window, labels]

            self.window.clear()
            return result, True

        scores = np.abs(self.model.score_samples(self.window.get().reshape(-1, 1)))
        labels = np.where(scores > self.score_threshold, 1, 0)

        result = np.c_[self.window.get(), labels]

        self.window.clear()
        return result, False
