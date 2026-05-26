from seqlens.models.baselines import (
    SUPPORTED_BASELINES,
    SUPPORTED_EVENT_BASELINES,
    baseline_forecast,
    event_majority_forecast,
    moving_average_forecast,
    naive_forecast,
)

__all__ = [
    "SUPPORTED_BASELINES",
    "SUPPORTED_EVENT_BASELINES",
    "baseline_forecast",
    "event_majority_forecast",
    "moving_average_forecast",
    "naive_forecast",
]
