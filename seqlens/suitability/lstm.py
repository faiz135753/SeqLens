from __future__ import annotations

from dataclasses import dataclass, field

from seqlens.diagnostics.timeseries import DiagnosticReport


@dataclass(frozen=True)
class SuitabilityReport:
    score: int
    verdict: str
    reasons: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"LSTM suitability score: {self.score}/100", f"Verdict: {self.verdict}"]
        if self.reasons:
            lines.append("Reasons:")
            lines.extend(f"- {reason}" for reason in self.reasons)
        if self.recommendations:
            lines.append("Recommendations:")
            lines.extend(f"- {item}" for item in self.recommendations)
        return "\n".join(lines)


def score_lstm_suitability(report: DiagnosticReport) -> SuitabilityReport:
    score = 50
    reasons: list[str] = []
    recommendations: list[str] = []

    if report.row_count >= 1000:
        score += 20
        reasons.append("The dataset has enough rows for a first LSTM experiment.")
    elif report.row_count >= 300:
        score += 10
        reasons.append("The dataset has a moderate number of rows.")
    else:
        score -= 20
        reasons.append("The dataset is short for deep learning.")
        recommendations.append("Start with naive, moving average, and tree-based baselines.")

    if report.is_regular_interval:
        score += 10
        reasons.append("The timestamps appear to have a regular interval.")
    else:
        score -= 10
        recommendations.append("Resample the data to a regular interval before training sequence models.")

    if report.target_missing_ratio == 0:
        score += 10
    elif report.target_missing_ratio <= 0.05:
        score -= 5
        recommendations.append("Impute or remove missing target values using train-only logic.")
    else:
        score -= 15
        recommendations.append("Resolve missing target values before model comparison.")

    if report.duplicate_timestamp_count:
        score -= 10
        recommendations.append("Aggregate or remove duplicate timestamps.")

    if report.lag1_autocorrelation is not None:
        autocorr = abs(report.lag1_autocorrelation)
        if autocorr >= 0.6:
            score += 15
            reasons.append("Strong lag-1 autocorrelation suggests sequence dependence.")
        elif autocorr >= 0.3:
            score += 8
            reasons.append("Moderate lag-1 autocorrelation is present.")
        else:
            score -= 10
            recommendations.append("Weak autocorrelation: verify that past values contain useful signal.")
    else:
        recommendations.append("Autocorrelation could not be estimated; inspect the target distribution.")

    if report.trend_strength is not None and report.trend_strength >= 0.4:
        score += 5
        reasons.append("A visible trend may provide learnable temporal structure.")

    if report.outlier_ratio > 0.1:
        score -= 10
        recommendations.append("High outlier ratio: inspect shocks, splits, and robust scaling.")
    elif report.outlier_ratio > 0.03:
        score -= 3
        recommendations.append("Inspect outliers before interpreting LSTM performance.")

    bounded_score = max(0, min(100, score))
    verdict = _verdict(bounded_score)

    if not recommendations:
        recommendations.append("Train LSTM only after comparing against simple baselines.")
    recommendations.append("Use validation metrics for optimization and reserve test data for final reporting.")

    return SuitabilityReport(
        score=bounded_score,
        verdict=verdict,
        reasons=reasons,
        recommendations=recommendations,
    )


def _verdict(score: int) -> str:
    if score >= 75:
        return "promising"
    if score >= 50:
        return "possible, but baseline comparison is required"
    return "not recommended until data or framing improves"

