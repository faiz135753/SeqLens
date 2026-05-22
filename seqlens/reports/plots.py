from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "seqlens-mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_actual_vs_predicted_plot(
    predictions: pd.DataFrame,
    *,
    path: str | Path,
    title: str,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = predictions.iloc[:, 0]
    ax.plot(x, predictions["actual"], marker="o", label="Actual")
    ax.plot(x, predictions["predicted"], marker="o", label="Predicted")
    ax.set_title(title)
    ax.set_xlabel(predictions.columns[0])
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
