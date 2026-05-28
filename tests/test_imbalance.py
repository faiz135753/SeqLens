import pandas as pd

from seqlens.automation import imbalance_diagnosis


def test_imbalance_diagnosis_classifies_event_definitions() -> None:
    support = pd.DataFrame(
        [
            {
                "threshold": 80,
                "horizon": 3,
                "split": "train",
                "support": 10000,
                "positive_support": 5,
            },
            {
                "threshold": 80,
                "horizon": 3,
                "split": "validation",
                "support": 3000,
                "positive_support": 2,
            },
            {
                "threshold": 80,
                "horizon": 3,
                "split": "test",
                "support": 3000,
                "positive_support": 3,
            },
            {
                "threshold": 80,
                "horizon": 12,
                "split": "train",
                "support": 10000,
                "positive_support": 300,
            },
            {
                "threshold": 80,
                "horizon": 12,
                "split": "validation",
                "support": 3000,
                "positive_support": 120,
            },
            {
                "threshold": 80,
                "horizon": 12,
                "split": "test",
                "support": 3000,
                "positive_support": 110,
            },
        ]
    )

    diagnosis = imbalance_diagnosis(support)

    severities = {
        (row.threshold, row.horizon): row.severity
        for row in diagnosis.itertuples(index=False)
    }
    assert severities[(80, 3)] == "unusable"
    assert severities[(80, 12)] == "imbalanced"
