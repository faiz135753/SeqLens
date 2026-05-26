from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LGBMEventModel:
    model: object
    feature_columns: list[str]


def train_lgbm_classifier(
    train_frame: pd.DataFrame,
    *,
    target_col: str,
    feature_columns: list[str],
    scale_pos_weight: float | None = None,
    random_state: int = 42,
) -> LGBMEventModel:
    try:
        from lightgbm import LGBMClassifier
    except (ImportError, OSError) as exc:
        raise ImportError(
            "LightGBM is not available. Install it with `pip install -e '.[lgbm]'`. "
            "On macOS, LightGBM may also require `brew install libomp`."
        ) from exc

    x_train = train_frame[feature_columns]
    y_train = train_frame[target_col].astype(int)
    if scale_pos_weight is None:
        scale_pos_weight = _scale_pos_weight(y_train)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    return LGBMEventModel(model=model, feature_columns=feature_columns)


def predict_lgbm_event_probability(model: LGBMEventModel, frame: pd.DataFrame) -> pd.Series:
    probabilities = model.model.predict_proba(frame[model.feature_columns])[:, 1]
    return pd.Series(probabilities, index=frame.index, name="probability")


def lgbm_feature_importance(model: LGBMEventModel) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": model.feature_columns,
            "importance": np.asarray(model.model.feature_importances_, dtype=float),
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


def _scale_pos_weight(target: pd.Series) -> float:
    positives = int((target == 1).sum())
    negatives = int((target == 0).sum())
    if positives == 0:
        return 1.0
    return float(negatives / positives)
