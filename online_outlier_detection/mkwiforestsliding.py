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
        return self._check_retrain_and_predict(h, slope, p_value)
