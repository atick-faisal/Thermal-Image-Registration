"""Metrics. One evaluation path: every method computes every metric through this code."""

from __future__ import annotations

from cmreg.metrics.registration import (
    EndPointError,
    MetricError,
    auc,
    corner_error,
    diagonal,
    endpoint_error,
    success_rate,
)
from cmreg.metrics.schema import RegistrationMetrics, auc_key, success_rate_key

__all__ = [
    "EndPointError",
    "MetricError",
    "RegistrationMetrics",
    "auc",
    "auc_key",
    "corner_error",
    "diagonal",
    "endpoint_error",
    "success_rate",
    "success_rate_key",
]
