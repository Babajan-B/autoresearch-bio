from __future__ import annotations

import os
from pathlib import Path

cache_root = Path(".cache")
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((cache_root / "matplotlib").resolve()))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, roc_curve


matplotlib.use("Agg")


PALETTE = {
    "ink": "#172033",
    "blue": "#2563eb",
    "blue_light": "#93c5fd",
    "cyan": "#0891b2",
    "gold": "#d97706",
    "rose": "#db2777",
    "grid": "#dbe4f0",
    "spine": "#94a3b8",
}


def _apply_publication_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "axes.edgecolor": PALETTE["spine"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "font.size": 10,
            "axes.titleweight": "bold",
        },
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(PALETTE["spine"])
    axis.spines["bottom"].set_color(PALETTE["spine"])
    axis.tick_params(length=0)


def _positive_probabilities(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.shape[1] == 2:
        return probabilities[:, 1]
    return probabilities.max(axis=1)


def save_roc_curve(y_true: np.ndarray, probabilities: np.ndarray, output_path: str | Path) -> Path:
    _apply_publication_theme()
    output = Path(output_path)
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    positive_prob = _positive_probabilities(probabilities)
    fpr, tpr, _ = roc_curve(y_true, positive_prob)
    roc_auc = auc(fpr, tpr)

    axis.fill_between(fpr, tpr, color=PALETTE["blue_light"], alpha=0.25)
    axis.plot(fpr, tpr, color=PALETTE["blue"], linewidth=2.6, label="Model")
    axis.plot([0, 1], [0, 1], linestyle=(0, (4, 4)), color=PALETTE["spine"], linewidth=1.3, label="Chance")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC Curve", pad=12)
    axis.text(
        0.97,
        0.08,
        f"AUROC = {roc_auc:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#eff6ff", "edgecolor": PALETTE["blue_light"]},
    )
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_path: str | Path,
) -> Path:
    _apply_publication_theme()
    output = Path(output_path)
    cm = confusion_matrix(y_true, y_pred)
    row_totals = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm.astype(float), row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)

    annotations = np.empty_like(cm, dtype=object)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            annotations[row, col] = f"{cm[row, col]}\n{normalized[row, col] * 100:.1f}%"

    figure, axis = plt.subplots(figsize=(5.8, 5.2))
    sns.heatmap(
        normalized,
        annot=annotations,
        fmt="",
        cmap=sns.light_palette(PALETTE["blue"], as_cmap=True),
        cbar=False,
        square=True,
        linewidths=1.0,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion Matrix", pad=12)
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def save_calibration_plot(y_true: np.ndarray, probabilities: np.ndarray, output_path: str | Path) -> Path:
    _apply_publication_theme()
    output = Path(output_path)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(6.4, 6.2),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.1], "hspace": 0.15},
    )
    reliability_axis, density_axis = axes
    if probabilities.ndim == 1:
        confidence = probabilities
        truth = y_true
    elif probabilities.shape[1] == 2:
        confidence = probabilities[:, 1]
        truth = y_true
    else:
        confidence = probabilities.max(axis=1)
        truth = (np.argmax(probabilities, axis=1) == y_true).astype(int)

    frac_pos, mean_pred = calibration_curve(truth, confidence, n_bins=8, strategy="quantile")
    reliability_axis.plot(
        mean_pred,
        frac_pos,
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        color=PALETTE["cyan"],
        label="Model",
    )
    reliability_axis.plot(
        [0, 1],
        [0, 1],
        linestyle=(0, (4, 4)),
        color=PALETTE["spine"],
        linewidth=1.2,
        label="Ideal",
    )
    reliability_axis.fill_between(mean_pred, frac_pos, alpha=0.15, color=PALETTE["cyan"])
    reliability_axis.set_xlim(0, 1)
    reliability_axis.set_ylim(0, 1)
    reliability_axis.set_title("Calibration Plot", pad=12)
    reliability_axis.set_ylabel("Observed frequency")
    reliability_axis.legend(frameon=False, loc="upper left")
    _style_axis(reliability_axis)

    density_axis.hist(confidence, bins=12, color=PALETTE["blue_light"], edgecolor="white")
    density_axis.set_xlabel("Predicted probability")
    density_axis.set_ylabel("Count")
    _style_axis(density_axis)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def save_shap_bar_plot(genes: list[str], values: np.ndarray, output_path: str | Path) -> Path:
    _apply_publication_theme()
    output = Path(output_path)
    top_n = min(len(genes), 20)
    genes = genes[:top_n]
    values = values[:top_n]
    figure, axis = plt.subplots(figsize=(8.2, max(4.5, top_n * 0.28)))
    palette = [PALETTE["blue"] if idx < 5 else PALETTE["blue_light"] for idx in range(top_n)]
    positions = np.arange(top_n)
    axis.hlines(positions, 0, values, color=palette, linewidth=2.2, alpha=0.9)
    axis.scatter(values, positions, color=palette, s=42, zorder=3)
    axis.set_yticks(positions)
    axis.set_yticklabels(genes)
    axis.invert_yaxis()
    axis.xaxis.grid(True, color=PALETTE["grid"])
    axis.yaxis.grid(False)
    for idx, value in enumerate(values):
        axis.text(value + max(values) * 0.015, idx, f"{value:.3f}", va="center", fontsize=8.6, color=PALETTE["ink"])
    axis.set_xlim(0, max(values) * 1.18 if len(values) else 1)
    axis.set_title("Top Genes by Mean Absolute SHAP Value", pad=12)
    axis.set_xlabel("Mean |SHAP|")
    axis.set_ylabel("Gene")
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output
