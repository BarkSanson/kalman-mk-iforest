import numpy as np
from pymannkendall import yue_wang_modification_test
from scipy.stats import wilcoxon

from .base_drift_detector import BaseDriftDetector


class MannKendallWilcoxonDriftDetector(BaseDriftDetector):
    def __init__(self, alpha: float, slope_threshold: float):
        self.alpha = alpha
        self.slope_threshold = slope_threshold

    def detect_drift(self, x: np.ndarray, y: np.ndarray) -> bool:
        _, h, _, _, _, _, _, slope, _ = \
            yue_wang_modification_test(x)
        d = np.around(x - y, decimals=3)
        stat, p_value = wilcoxon(d, zero_method='zsplit')

        return (h and slope > self.slope_threshold) or p_value < self.alpha
