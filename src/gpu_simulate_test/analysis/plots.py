from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cdf(
    *,
    real: pd.Series,
    sim: pd.Series,
    out_path: Path,
    title: str,
    xlabel: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _cdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        x = np.sort(values.dropna().to_numpy(dtype=float))
        if x.size == 0:
            return x, x
        y = np.arange(1, x.size + 1) / x.size
        return x, y

    x_r, y_r = _cdf(real)
    x_s, y_s = _cdf(sim)

    plt.figure(figsize=(6, 4))
    if x_r.size:
        plt.plot(x_r, y_r, label="real")
    if x_s.size:
        plt.plot(x_s, y_s, label="sim")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

