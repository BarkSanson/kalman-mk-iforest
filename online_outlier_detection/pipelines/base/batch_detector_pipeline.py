import numpy as np

from .base_detector_pipeline import BaseDetectorPipeline
from online_outlier_detection.window.batch_window import BatchWindow


class BatchDetectorPipeline(BaseDetectorPipeline):
    def __init__(self,
                 score_threshold: float,
                 alpha: float,
                 slope_threshold: float,
                 window_size: int):
        super().__init__(score_threshold, alpha, slope_threshold, window_size)

        self.window = BatchWindow(self.window_size)

    def _first_training(self):
        scores, labels = super()._first_training()
        self.window.clear()

        return scores, labels

    def update(self, x) -> tuple[np.ndarray, np.ndarray] | None:
        pass
