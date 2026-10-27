"""Draw the theory-audited SERVO execution pipeline as editable vector PDF."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "fig_algorithm_v2"

TEAL = "#007C83"
BLUE = "#377EB8"
ORANGE = "#E69F00"
PURPLE = "#7B61A8"
INK = "#263238"
MUTED = "#546E7A"
LIGHT = "#F4F7F8"


def box(ax, x, y, w, h, title, lines, color, tag):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15, edgecolor=color, facecolor="white", zorder=2,
    )
    ax.add_patch(patch)
    ax.add_patch(FancyBboxPatch(
        (x, y + h - 0.16), w, 0.16,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0, facecolor=color, zorder=3,
    ))
    ax.text(x + 0.025, y + h - 0.08, tag, ha="left", va="center",
            color="white", fontsize=7.0, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h - 0.245, title, ha="center", va="center",
            color=INK, fontsize=7.8, fontweight="bold", zorder=4)
    yy = y + h - 0.365
    for line in lines:
        ax.text(x + w / 2, yy, line, ha="center", va="top", color=MUTED,
                fontsize=6.35, linespacing=1.15, zorder=4)
        yy -= 0.088


def arrow(ax, x1, y1, x2, y2, color=INK, rad=0.0, lw=1.1, style="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, linestyle=style, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=1,
    ))


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.15, 2.68))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.018, 0.268, 0.518, 0.768]
    w, y, h = 0.214, 0.25, 0.68
    box(ax, xs[0], y, w, h, "Coverage estimation", [
        r"Bipartite flow--event GNN",
        r"Predict $\widehat q_{k,t}(\omega)$",
        r"Predict $\widehat{imp}_t(\omega)$",
        "Mobility, link, load, context",
    ], TEAL, "1  SENSE")
    box(ax, xs[1], y, w, h, "Fixed-quality selection", [
        r"$P_{sel}$: submodular greedy",
        r"Cardinality: $1-1/e$",
        r"Matroid greedy: $1/2$",
        "Projection: no ratio claimed",
    ], BLUE, "2  SELECT")
    box(ax, xs[2], y, w, h, "Coupled allocation", [
        r"$P_{coup}$: residual SAC",
        "Value-aware feasible reference",
        "Exact budget normalization",
        "No global-optimality claim",
    ], ORANGE, "3  ALLOCATE")
    box(ax, xs[3], y, w, h, "Execute and adapt", [
        "Latency, SLA, energy, completion",
        "Prediction-error feedback",
        "Page--Hinkley drift test",
        "Refit tier-wise calibration",
    ], PURPLE, "4  OBSERVE")

    cy = y + h / 2
    for i in range(3):
        arrow(ax, xs[i] + w + 0.004, cy, xs[i + 1] - 0.006, cy)

    # Feedback arc stays outside the content boxes.
    arrow(ax, xs[3] + w * 0.75, 0.19, xs[0] + w * 0.25, 0.19,
          color=PURPLE, rad=0.0, lw=1.05)
    ax.text(0.50, 0.205, "delayed task outcomes and prediction-error feedback",
            ha="center", va="center", fontsize=7.0, color=PURPLE)

    band = FancyBboxPatch((0.018, 0.035), 0.964, 0.12,
                          boxstyle="round,pad=0.01,rounding_size=0.012",
                          linewidth=0.7, edgecolor="#B0BEC5", facecolor=LIGHT)
    ax.add_patch(band)
    ax.text(0.035, 0.095, "Guarantee boundary", ha="left", va="center",
            fontsize=6.9, fontweight="bold", color=INK)
    ax.text(0.235, 0.112,
            "Approximation/no-regret results cover fixed-quality ideal algorithms only;",
            ha="left", va="center", fontsize=6.1, color=MUTED)
    ax.text(0.235, 0.075,
            "resource coupling, projection, SAC, and PH adaptation are evaluated empirically.",
            ha="left", va="center", fontsize=6.1, color=MUTED)

    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
