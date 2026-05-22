from pathlib import Path

import pandas as pd

from seqlens.experiments import ExperimentConfig, run_naive_baseline


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

    predictions = pd.read_csv(result.run_dir / "test_predictions.csv")
    assert list(predictions.columns) == ["date", "actual", "predicted", "error"]
    assert len(predictions) > 0
