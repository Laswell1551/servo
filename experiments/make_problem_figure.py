#!/usr/bin/env python3
"""Draw the revised system/problem map as editable vector PDF."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GRAY = "#666666"
LIGHT = "#F2F2F2"
INK = "#222222"


def clean(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(ax, x, y, w, h, text, fc="white", ec=GRAY, fs=7.0, lw=1.0, weight="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, weight=weight, linespacing=1.15)
    return p


def arrow(ax, p0, p1, color=GRAY, style="-|>", lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9,
                                linewidth=lw, linestyle=ls, color=color,
                                shrinkA=1, shrinkB=1))


def panel_scene(ax):
    clean(ax)
    ax.text(0.02, 0.96, "(a) Dynamic V2X continuum", fontsize=8.2,
            weight="bold", va="top")
    # stylized roads
    ax.add_patch(Rectangle((0.02, 0.39), 0.92, 0.18, fc="#E5E7E9", ec="none"))
    ax.add_patch(Rectangle((0.39, 0.08), 0.18, 0.80, fc="#E5E7E9", ec="none"))
    ax.plot([0.02, 0.94], [0.48, 0.48], color="white", lw=1.0, ls=(0, (5, 4)))
    ax.plot([0.48, 0.48], [0.08, 0.88], color="white", lw=1.0, ls=(0, (5, 4)))
    nodes = [(0.17, 0.69), (0.79, 0.69), (0.17, 0.25), (0.79, 0.25)]
    for j, (x, y) in enumerate(nodes):
        ax.add_patch(Rectangle((x - 0.025, y - 0.045), 0.05, 0.09, fc=BLUE, ec="white", lw=0.6))
        ax.text(x, y - 0.075, f"edge {j+1}", fontsize=5.8, ha="center", color=BLUE)
    vehicles = [(0.13, 0.47), (0.33, 0.47), (0.48, 0.22), (0.48, 0.74), (0.72, 0.47)]
    for k, (x, y) in enumerate(vehicles):
        ax.add_patch(FancyBboxPatch((x - 0.035, y - 0.022), 0.07, 0.044,
                                    boxstyle="round,pad=0.004,rounding_size=0.008",
                                    fc=GREEN, ec="white", lw=0.5))
        ax.text(x, y, f"$k_{k+1}$", fontsize=5.5, color="white", ha="center", va="center")
    events = [(0.43, 0.49), (0.52, 0.42), (0.67, 0.52)]
    for w, (x, y) in enumerate(events):
        ax.add_patch(Circle((x, y), 0.025, fc=RED if w == 0 else ORANGE, ec="white", lw=0.6))
        ax.text(x, y + 0.045, rf"$\omega_{w+1}$", fontsize=6.0, ha="center")
    for p0, p1 in [((0.13, .47), events[0]), ((.33, .47), events[0]),
                   ((.48, .22), events[1]), ((.48, .74), events[2]), ((.72, .47), events[2])]:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GREEN, lw=0.9, alpha=.9)
    ax.text(0.49, 0.025, "mobility, interference, heterogeneous edges,\nand dynamic requests",
            fontsize=5.8, color=GRAY, ha="center", va="bottom")


def panel_graph(ax):
    clean(ax)
    ax.text(0.02, 0.96, "(b) Coverage graph", fontsize=8.2,
            weight="bold", va="top")
    yf = [0.78, 0.64, 0.50, 0.36, 0.22]
    ye = [0.72, 0.50, 0.28]
    edges = [(0, 0, 2.0), (1, 0, 1.2), (2, 1, 1.8), (3, 1, 1.0), (3, 2, 1.2), (4, 2, 1.7)]
    for f, e, lw in edges:
        color = RED if e == 0 and f < 2 else BLUE
        ax.plot([0.17, 0.62], [yf[f], ye[e]], color=color, lw=lw)
    for f, y in enumerate(yf):
        ax.add_patch(Circle((0.17, y), 0.018, fc=GREEN, ec="white", lw=.5))
        ax.text(0.12, y, f"$k_{f+1}$", fontsize=6.6, ha="right", va="center")
    for e, y in enumerate(ye):
        ax.add_patch(Circle((0.62, y), 0.020, fc=ORANGE, ec="white", lw=.5))
        ax.text(0.68, y, ["hazard", "pedestrian", "merge"][e], fontsize=6.2, ha="left", va="center")
    ax.text(0.43, 0.84, "redundant reports", fontsize=5.8,
            ha="center", color=RED)
    box(ax, 0.09, 0.02, 0.82, 0.12,
        r"$V(S)=\sum_{\omega\in\Omega}{\rm imp}(\omega)\max_{k\in S:\,\omega\in{\rm cov}(k)}q_k(\omega)$",
        fc="#F7FBFF", ec=BLUE, fs=7.3)


def panel_problem(ax):
    clean(ax)
    ax.text(0.02, 0.96, "(c) Problem and guarantee map", fontsize=8.2,
            weight="bold", va="top")
    box(ax, 0.08, 0.76, 0.84, 0.13,
        "P1: joint discrete--continuous orchestration\nresource-dependent quality + coupled interference",
        fc="#FFF4EE", ec=RED, fs=5.9, weight="bold")
    arrow(ax, (0.50, 0.76), (0.50, 0.68), color=GRAY)
    box(ax, 0.03, 0.47, 0.44, 0.18,
        r"$P_{\mathrm{sel}}$: fixed $q$" "\n"
        r"continuous greedy: $1-1/e$" "\n"
        r"lazy: $1/2$; cardinality: $1-1/e$",
        fc="#EEF8F4", ec=GREEN, fs=5.25)
    box(ax, 0.53, 0.47, 0.44, 0.18,
        r"$P_{\mathrm{fix}}$: fixed interference" "\n"
        "convex in time variables\nshared budgets couple flows",
        fc="#F0F7FC", ec=BLUE, fs=5.25)
    arrow(ax, (0.50, 0.47), (0.50, 0.39), color=GRAY)
    box(ax, 0.15, 0.17, 0.70, 0.18,
        r"$P_{\mathrm{coup}}$: coupled interference (nonconvex)" "\n"
        "SAC residual allocator + hard budget projection\nvalidated empirically; no approximation claim",
        fc="#FFF9E8", ec=ORANGE, fs=5.6, weight="bold")
    ax.text(0.50, 0.055,
            "Resource projection under binding budgets is outside the stated selection guarantee.",
            fontsize=5.3, color=RED, ha="center")


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.75),
                             gridspec_kw={"wspace": 0.10, "width_ratios": [1.0, 1.0, 1.28]})
    panel_scene(axes[0]); panel_graph(axes[1]); panel_problem(axes[2])
    fig.savefig(OUT / "fig_system_model_v2.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig_system_model_v2.png", bbox_inches="tight", facecolor="white", dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()
