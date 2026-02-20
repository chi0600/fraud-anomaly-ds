from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass
class ThresholdMetrics:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float


def metrics_at_threshold(y_true, y_prob, threshold: float) -> ThresholdMetrics:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return ThresholdMetrics(
        threshold=float(threshold),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        precision=float(precision),
        recall=float(recall),
    )


def scan_thresholds(y_true, y_prob, thresholds: Iterable[float]) -> List[ThresholdMetrics]:
    return [metrics_at_threshold(y_true, y_prob, t) for t in thresholds]