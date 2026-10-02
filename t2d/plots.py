"""Figures for the RFECV + stacking experiment."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve  # noqa: E402


def plot_rfecv_curve(rfecv, path):
    res = rfecv.cv_results_
    n = np.arange(rfecv.min_features_to_select, rfecv.min_features_to_select + len(res["mean_test_score"]))
    mean, std = res["mean_test_score"], res["std_test_score"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n, mean, marker="o", color="#2a6fdb", label="mean CV score")
    ax.fill_between(n, mean - std, mean + std, color="#2a6fdb", alpha=0.15, label="± 1 std")
    ax.axvline(rfecv.n_features_, color="#d1495b", linestyle="--", label=f"selected = {rfecv.n_features_}")
    ax.set_xlabel("Number of features")
    ax.set_ylabel(f"CV {rfecv.scoring}")
    ax.set_title("Recursive Feature Elimination with Cross-Validation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_ranking(ranking, path):
    """Horizontal bar chart of RFECV ranks (1 = selected)."""
    ranking = ranking.sort_values(ascending=False)
    colors = ["#2a6fdb" if r == 1 else "#b0b7c3" for r in ranking.values]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(ranking) + 1.2))
    ax.barh(ranking.index, ranking.values, color=colors)
    ax.set_xlabel("RFECV rank (1 = selected)")
    ax.set_title("Feature ranking")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(curves, path):
    """curves: {label: (y_true, y_score, auc)}"""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for label, (y_true, y_score, auc) in sorted(curves.items(), key=lambda kv: -kv[1][2]):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        lw = 2.5 if label.startswith("stacking") else 1.2
        ax.plot(fpr, tpr, lw=lw, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the hold-out test set")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Non-diabetic", "T2DM"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
