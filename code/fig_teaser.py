#!/usr/bin/env python3
"""fig_teaser.py — From bit-fair to value-fair orchestration (SERVO Fig. 1).
Two-panel flat vector schematic; exact labels; black-and-white legible."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch
import numpy as np

import os
OUT = os.path.dirname(os.path.abspath(__file__))
TEAL = "#1b9e77"; GREY = "#9e9e9e"; RED = "#d62728"
ROAD = "#ececec"; INK = "#222222"; HALO = "#bfe6d8"

plt.rcParams.update({"font.size": 8, "font.family": "Times New Roman",
                     "mathtext.fontset": "stix", "savefig.dpi": 300,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# ---------- glyph helpers ----------
def car(ax, x, y, idle=False, s=0.62, cover=None):
    body_c = "#ffffff" if idle else "#dfeee8"
    ec = GREY if idle else TEAL if cover else INK
    lw = 1.0 if idle else 1.4
    ax.add_patch(FancyBboxPatch((x-0.55*s, y-0.16*s), 1.10*s, 0.30*s,
                boxstyle="round,pad=0.015,rounding_size=0.06", fc=body_c, ec=ec, lw=lw, zorder=4))
    ax.add_patch(FancyBboxPatch((x-0.26*s, y+0.06*s), 0.52*s, 0.20*s,
                boxstyle="round,pad=0.01,rounding_size=0.05", fc=body_c, ec=ec, lw=lw, zorder=4))
    for dx in (-0.30, 0.30):
        ax.add_patch(Circle((x+dx*s, y-0.18*s), 0.085*s, fc="#444", ec="#444", zorder=5))

def halo(ax, x, y, r=0.62):
    ax.add_patch(Circle((x, y), r, fc=HALO, ec="none", alpha=0.9, zorder=1))

def hazard(ax, x, y, covered, s=0.6):
    if covered: halo(ax, x, y, 0.62)
    tri = Polygon([[x, y+0.46*s], [x-0.46*s, y-0.34*s], [x+0.46*s, y-0.34*s]],
                  closed=True, fc=RED if covered else "#f2c9c9",
                  ec=RED, lw=2.0, zorder=3)
    ax.add_patch(tri)
    ax.text(x, y-0.02*s, "!", ha="center", va="center", fontsize=11,
            fontweight="bold", color="white" if covered else RED, zorder=4)

def pedestrian(ax, x, y, covered, s=0.6):
    c = TEAL if covered else GREY
    if covered: halo(ax, x, y, 0.58)
    ax.add_patch(Circle((x, y+0.30*s), 0.12*s, fc=c, ec=c, zorder=3))
    ax.add_patch(Polygon([[x, y+0.20*s], [x-0.17*s, y-0.30*s], [x+0.17*s, y-0.30*s]],
                 closed=True, fc=c, ec=c, zorder=3))

def merge(ax, x, y, covered, s=0.6):
    c = TEAL if covered else GREY
    if covered: halo(ax, x, y, 0.58)
    for dx in (-0.38, 0.38):
        ax.add_patch(FancyArrowPatch((x+dx*s, y-0.34*s), (x, y+0.06*s),
                     arrowstyle="-|>", mutation_scale=10, color=c, lw=1.8, zorder=3))
    ax.add_patch(FancyArrowPatch((x, y+0.06*s), (x, y+0.40*s),
                 arrowstyle="-|>", mutation_scale=10, color=c, lw=1.8, zorder=3))

def link(ax, car_xy, ev_xy, redundant):
    style = dict(arrowstyle="-|>", mutation_scale=11,
                 shrinkA=10, shrinkB=14, zorder=2)
    if redundant:
        ax.add_patch(FancyArrowPatch(car_xy, ev_xy, color=GREY, lw=1.2,
                     linestyle=(0, (4, 3)), **style))
    else:
        ax.add_patch(FancyArrowPatch(car_xy, ev_xy, color=TEAL, lw=2.0, **style))

def intersection(ax):
    ax.add_patch(plt.Rectangle((0.4, 4.05), 9.2, 1.25, fc=ROAD, ec="none", zorder=0))
    ax.add_patch(plt.Rectangle((4.35, 0.6), 1.30, 8.0, fc=ROAD, ec="none", zorder=0))
    for xx in np.arange(0.7, 9.6, 0.7):
        ax.plot([xx, xx+0.35], [4.675, 4.675], color="#ffffff", lw=1.0, zorder=0)
    for yy in np.arange(0.9, 8.5, 0.7):
        ax.plot([5.0, 5.0], [yy, yy+0.35], color="#ffffff", lw=1.0, zorder=0)

def panel_frame(ax, title, subtitle):
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 9.7, 9.7,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                fc="white", ec="#cfcfcf", lw=1.2, zorder=0))
    ax.add_patch(FancyBboxPatch((0.15, 8.7), 9.7, 1.15,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                fc="#f6f6f6", ec="#cfcfcf", lw=1.2, zorder=0))
    ax.text(5, 9.45, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK, zorder=6)
    ax.text(5, 8.95, subtitle, ha="center", va="center", fontsize=7.5,
            color="#555", style="italic", zorder=6)

# shared scene coordinates
HAZ, PED, MRG = (5.0, 7.3), (8.05, 4.65), (1.95, 4.65)
CARS = [(3.0, 2.2), (5.0, 1.95), (7.0, 2.2), (2.6, 6.5), (7.4, 6.5)]

def setup(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); ax.set_aspect("equal")

# ---------- figure ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.3, 3.7))
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.10, wspace=0.22)

# ===== LEFT: weight-fair / value-blind =====
setup(axL); panel_frame(axL, "Weight-fair  (value-blind)", "bits / throughput drive allocation")
intersection(axL)
hazard(axL, *HAZ, covered=True)          # the one over-served event
pedestrian(axL, *PED, covered=False)     # uncovered
merge(axL, *MRG, covered=False)          # uncovered
# cars: 3 clustered all report the same hazard (redundant); 2 idle
for i, c in enumerate(CARS):
    idle = i in (3, 4)
    car(axL, *c, idle=idle)
for i in (0, 1, 2):
    link(axL, CARS[i], HAZ, redundant=True)
axL.text(5, -0.55, "budget wasted on redundant reports of one event",
         ha="center", va="center", fontsize=7.2, color="#555")

# ===== RIGHT: SERVO / value-fair =====
setup(axR); panel_frame(axR, "SERVO  (value-fair)", "semantic value drives allocation")
intersection(axR)
hazard(axR, *HAZ, covered=True)
pedestrian(axR, *PED, covered=True)
merge(axR, *MRG, covered=True)
# each distinct event covered by exactly one car; 2 idle
cover_map = {1: HAZ, 4: PED, 3: MRG}
for i, c in enumerate(CARS):
    car(axR, *c, idle=(i in (0, 2)), cover=(i in cover_map))
for i, ev in cover_map.items():
    link(axR, CARS[i], ev, redundant=False)
axR.text(5, -0.55, "budget spread to cover distinct events",
         ha="center", va="center", fontsize=7.2, color="#555")

# ===== center arrow + label =====
fig.patches.append(FancyArrowPatch((0.485, 0.55), (0.515, 0.55),
    transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22,
    color=INK, lw=2.4, zorder=10))
fig.text(0.5, 0.62, "from bit-fair\nto value-fair", ha="center", va="bottom",
         fontsize=7.5, fontweight="bold", color=INK)

# ===== legend =====
import matplotlib.lines as mlines
hcov = mlines.Line2D([], [], marker="o", ms=8, mfc=TEAL, mec=TEAL, ls="",
                     label="covered (distinct meaning)")
hred = mlines.Line2D([], [], color=GREY, ls=(0, (4, 3)), lw=1.4,
                     label="redundant report")
hunc = mlines.Line2D([], [], marker="o", ms=8, mfc="white", mec=GREY, ls="",
                     label="uncovered event")
fig.legend(handles=[hcov, hred, hunc], loc="lower center", ncol=3,
           frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01),
           handletextpad=0.4, columnspacing=1.4)

fig.savefig(f"{OUT}/fig_teaser.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_teaser.png", bbox_inches="tight")
print("saved fig_teaser.pdf / .png")
