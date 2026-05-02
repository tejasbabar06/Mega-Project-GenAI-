"""
utils/visualization.py
-----------------------
Generates matplotlib charts and returns them as base64-encoded PNG strings
so they can be embedded directly in HTML via <img src="data:image/png;base64,...">.

Charts
------
  missing_values_chart   – bar chart of columns with missing values
  feature_importance_chart – horizontal bar chart of feature scores
  pca_variance_chart     – bar chart of PCA explained variance ratios

All charts use a dark theme that matches the app's UI (dark background,
accent blues/purples/greens).
"""

import io
import base64

import matplotlib
matplotlib.use("Agg")                        # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Shared styling constants (match the web app's CSS tokens)
# ══════════════════════════════════════════════════════════════════════

_BG_DARK   = "#0a0e1a"
_BG_CARD   = "#111827"
_TEXT_PRI  = "#e8eaf6"
_TEXT_SEC  = "#8892b0"
_BLUE      = "#6382ff"
_PURPLE    = "#a855f7"
_CYAN      = "#22d3ee"
_GREEN     = "#34d399"
_ORANGE    = "#fb923c"
_GRID      = (99/255, 130/255, 255/255, 0.08)


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _apply_dark_style(ax, fig):
    """Apply the dark theme to an axes and figure."""
    fig.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_CARD)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_TEXT_SEC)
    ax.spines["bottom"].set_color(_TEXT_SEC)
    ax.tick_params(colors=_TEXT_SEC, labelsize=9)
    ax.xaxis.label.set_color(_TEXT_PRI)
    ax.yaxis.label.set_color(_TEXT_PRI)
    ax.title.set_color(_TEXT_PRI)


class VisualizationUtils:
    """
    Collection of static chart generators.
    Each method returns a base64-encoded PNG string.
    """

    # ------------------------------------------------------------------
    # 1. Missing Values Bar Chart
    # ------------------------------------------------------------------

    @staticmethod
    def missing_values_chart(missing_dict: dict) -> str:
        """
        Vertical bar chart showing missing-value counts per column.

        Parameters
        ----------
        missing_dict : {column_name: missing_count, ...}

        Returns
        -------
        Base64 PNG string (empty string if no missing values).
        """
        if not missing_dict:
            return ""

        cols   = list(missing_dict.keys())
        counts = list(missing_dict.values())

        fig, ax = plt.subplots(figsize=(max(5, len(cols) * 0.8), 4))
        _apply_dark_style(ax, fig)

        # Gradient-like colours from orange → red based on count
        max_c = max(counts) if counts else 1
        colors = [
            plt.cm.YlOrRd(0.3 + 0.6 * (c / max_c)) for c in counts
        ]

        bars = ax.bar(cols, counts, color=colors, edgecolor="none",
                       width=0.6, zorder=3)

        # Value labels on top of each bar
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=_ORANGE)

        ax.set_ylabel("Missing Count", fontsize=10, fontweight="bold")
        ax.set_title("Missing Values per Column", fontsize=12, fontweight="bold",
                      pad=12)
        ax.grid(axis="y", color=_GRID, linestyle="--", alpha=0.5, zorder=0)

        plt.xticks(rotation=35, ha="right", fontsize=8)
        plt.tight_layout()

        return _fig_to_base64(fig)

    # ------------------------------------------------------------------
    # 2. Feature Importance Chart
    # ------------------------------------------------------------------

    @staticmethod
    def feature_importance_chart(feature_scores: dict) -> str:
        """
        Horizontal bar chart showing feature importance scores.

        Parameters
        ----------
        feature_scores : {feature_name: score, ...}

        Returns
        -------
        Base64 PNG string (empty string if no scores).
        """
        if not feature_scores:
            return ""

        # Sort by score descending
        sorted_items = sorted(feature_scores.items(), key=lambda x: x[1],
                              reverse=True)
        names  = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]

        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.4)))
        _apply_dark_style(ax, fig)

        # Gradient colours from green (high) to blue (low)
        max_s = max(scores) if scores else 1
        colors = [
            plt.cm.cool(0.2 + 0.7 * (s / max_s)) if max_s > 0 else _BLUE
            for s in scores
        ]

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, scores, color=colors, edgecolor="none",
                        height=0.6, zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()  # highest score on top

        # Value labels
        for bar, val in zip(bars, scores):
            ax.text(bar.get_width() + max_s * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", ha="left", va="center",
                    fontsize=8, color=_TEXT_SEC)

        ax.set_xlabel("Importance Score", fontsize=10, fontweight="bold")
        ax.set_title("Feature Importance Ranking", fontsize=12,
                      fontweight="bold", pad=12)
        ax.grid(axis="x", color=_GRID, linestyle="--", alpha=0.5, zorder=0)

        plt.tight_layout()
        return _fig_to_base64(fig)

    # ------------------------------------------------------------------
    # 3. PCA Explained Variance Chart
    # ------------------------------------------------------------------

    @staticmethod
    def pca_variance_chart(variance_ratios: list) -> str:
        """
        Bar chart of PCA explained variance ratios with cumulative line.

        Parameters
        ----------
        variance_ratios : list of floats (one per component)

        Returns
        -------
        Base64 PNG string.
        """
        if not variance_ratios:
            return ""

        n = len(variance_ratios)
        components = [f"PC{i+1}" for i in range(n)]
        percentages = [v * 100 for v in variance_ratios]
        cumulative  = np.cumsum(percentages)

        fig, ax = plt.subplots(figsize=(max(5, n * 0.9), 4))
        _apply_dark_style(ax, fig)

        # Bars for individual variance
        bars = ax.bar(components, percentages, color=_PURPLE,
                       edgecolor="none", width=0.55, zorder=3,
                       alpha=0.85, label="Individual")

        # Cumulative line
        ax.plot(components, cumulative, color=_CYAN, marker="o",
                markersize=6, linewidth=2, zorder=4, label="Cumulative")

        # Value labels
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{pct:.1f}%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=_PURPLE)

        ax.set_ylabel("Explained Variance (%)", fontsize=10, fontweight="bold")
        ax.set_title("PCA — Explained Variance Ratio", fontsize=12,
                      fontweight="bold", pad=12)
        ax.grid(axis="y", color=_GRID, linestyle="--", alpha=0.5, zorder=0)
        ax.legend(fontsize=8, facecolor=_BG_CARD, edgecolor=_TEXT_SEC,
                  labelcolor=_TEXT_PRI)

        plt.tight_layout()
        return _fig_to_base64(fig)
