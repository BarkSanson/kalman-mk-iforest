import numpy as np
from filterpy.kalman import KalmanFilter

from online_outlier_detection.pipelines.base.base_detector_pipeline import BaseDetectorPipeline


class KalmanBasedDetectorPipeline(BaseDetectorPipeline):
    def __init__(self,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int):
        super().__init__(score_threshold, alpha, slope_threshold, window_size)

        self.kf = KalmanFilter(dim_x=1, dim_z=1)
        self.kf.Q = 0.001
        self.kf.F = np.array([[1]])
        self.kf.H = np.array([[1]])
        self.kf.x = np.array([0])
        self.kf.P = np.array([1])

        self.filtered_window = None # Either Batch or Sliding
        self.filtered_reference_window = np.array([])

    def update(self, x) -> tuple[np.ndarray, np.ndarray] | None:
        pass

    def _retrain(self):
        super()._retrain()
        self.filtered_reference_window = self.filtered_window.get().copy()
