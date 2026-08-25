"""
Plotting utilities: episode profiles and cross-controller comparison plots.
Uses matplotlib only (no display backend required).
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_profile(time_s: List[float], series: Dict[str, List[float]], ylabel: str,
                  title: str, out_path: str) -> None:
    """Plot one or more time-series (e.g. multiple controllers) on one axes."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, values in series.items():
        ax.plot(time_s, values, label=label, linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison_bar(labels: List[str], means: List[float], stds: List[float],
                         ylabel: str, title: str, out_path: str) -> None:
    """Bar chart with error bars for cross-controller metric comparison."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = range(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
