from pathlib import Path

import pandas as pd

from seqlens.experiments import (
    ExperimentConfig,
    run_baseline,
    run_baseline_comparison,
    run_naive_baseline,
)


def test_run_naive_baseline_writes_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig(
        data_path=str(root / "examples" / "sample_stock.csv"),
        time_col="date",
        target_col="close",
        validation_size=0.25,
        test_size=0.25,
    )

    result = run_naive_baseline(config, output_dir=tmp_path)

    assert result.run_dir.exists()
    assert (result.run_dir / "config.yaml").exists()
    assert (result.run_dir / "metrics.json").exists()
    assert (result.run_dir / "validation_predictions.csv").exists()
    assert (result.run_dir / "test_predictions.csv").exists()
    assert (result.run_dir / "validation_actual_vs_predicted.png").exists()
    assert (result.run_dir / "test_actual_vs_predicted.png").exists()
    assert (result.run_dir / "report.md").exists()

    predictions = pd.read_csv(result.run_dir / "test_predictions.csv")
    assert list(predictions.columns) == ["date", "actual", "predicted", "error"]
    assert len(predictions) > 0

    report = (result.run_dir / "report.md").read_text(encoding="utf-8")
    assert "# SeqLens Baseline Report" in report
    assert "Validation Metrics" in report
    assert "Test Metrics" in report


def test_run_moving_average_baseline_writes_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig(
        data_path=str(root / "examples" / "sample_stock.csv"),
        time_col="date",
        target_col="close",
        moving_average_window=3,
        validation_size=0.25,
        test_size=0.25,
    )

    result = run_baseline(config, model_name="moving_average", output_dir=tmp_path)

    assert result.run_dir.name.endswith("_moving_average")
    assert (result.run_dir / "report.md").exists()
    assert (result.run_dir / "metrics.json").exists()

    report = (result.run_dir / "report.md").read_text(encoding="utf-8")
    assert "| Model | moving_average |" in report


def test_run_baseline_comparison_writes_comparison_table(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig(
        data_path=str(root / "examples" / "sample_stock.csv"),
        time_col="date",
        target_col="close",
        models=["naive", "moving_average", "lstm"],
        validation_size=0.25,
        test_size=0.25,
    )

    result = run_baseline_comparison(config, output_dir=tmp_path)

    assert result.comparison_path.exists()
    assert result.report_path.exists()
    assert len(result.runs) == 2

    comparison = pd.read_csv(result.comparison_path)
    assert comparison["model"].tolist() == ["naive", "moving_average"]
    assert "validation_rmse" in comparison.columns
    assert "test_rmse" in comparison.columns

    report = result.report_path.read_text(encoding="utf-8")
    assert "# SeqLens Baseline Comparison Report" in report
    assert "Best Validation Baseline" in report
    assert "Comparison Table" in report
