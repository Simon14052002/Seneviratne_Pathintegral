"""
Plot the reproduction of Fig. 5b (exact solid lines) of
Seneviratne, Walters & Wang, ACS Omega 2024, 9, 9666,
from the saved path-integral results.

Usage:  python3 read_figure5b.py [tag]          (default: main)
Produces figure5b_reproduction.png and, if reference data are present,
figure5b_validation.png (populations + per-comparison deviation panel;
each comparison has its own colour - no same-colour overlays).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

tag = sys.argv[1] if len(sys.argv) > 1 else "main"
data = np.load(f"figure5b_{tag}.npz")
t = data["t_list"]
pops = data["pop_direct"]          # populations from L(t) vec(rho0)

site_colors = ["red", "blue", "green", "black"]

# ---- main figure: same style as the target panel --------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for i in range(4):
    ax.plot(t, pops[:, i], color=site_colors[i], lw=1.8,
            label=f"Exact-Site {i + 1}")
ax.set_xlabel("time/fs", fontsize=13)
ax.set_ylabel("Population", fontsize=13)
ax.set_xlim(-25, 1025)
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=10, loc="upper right")
ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig("figure5b_reproduction.png", dpi=200)
print("wrote figure5b_reproduction.png")

# ---- validation figure: populations + deviation panel ----------------------
comparisons = []   # (label, color, t_grid, max-over-sites |Delta pop|)


def add_comparison(label, color, t2, p2):
    p2i = np.array([np.interp(t, t2, p2[:, i]) for i in range(4)]).T
    comparisons.append((label, color, t, np.abs(pops - p2i).max(axis=1)))


if os.path.exists("heom_Nk3_d5.npz"):
    h = np.load("heom_Nk3_d5.npz")
    add_comparison("vs HEOM (exact reference)", "tab:purple",
                   h["t_list_fs"], h["pops"])
for other, label, color in [
        ("eps1e5", "vs eps = 1e-5 (SVD cutoff x10)", "tab:orange"),
        ("mem350", "vs 350 fs memory (+100 fs)", "tab:cyan"),
        ("dt5", "vs dt = 5 fs (Trotter/2)", "tab:brown"),
        ("dt5b", "vs dt = 5 fs (Trotter/2)", "tab:brown")]:
    f = f"figure5b_{other}.npz"
    if os.path.exists(f) and other != tag:
        d = np.load(f)
        add_comparison(label, color, d["t_list"], d["pop_direct"])

if comparisons:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True,
                                   height_ratios=[2.2, 1.0])
    for i in range(4):
        ax1.plot(t, pops[:, i], color=site_colors[i], lw=1.8,
                 label=f"Site {i + 1}")
    ax1.set_ylabel("Population", fontsize=12)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=9, loc="upper right",
               title=f"path integral ({tag})", title_fontsize=9)
    seen = set()
    for label, color, tc, dev in comparisons:
        if label in seen:
            continue
        seen.add(label)
        ax2.semilogy(tc, np.maximum(dev, 1e-8), color=color, lw=1.4,
                     label=label)
    ax2.set_xlabel("time/fs", fontsize=12)
    ax2.set_ylabel(r"max$_i$ |$\Delta$ population|", fontsize=11)
    ax2.set_xlim(-25, 1025)
    ax2.set_ylim(1e-6, 3e-2)
    ax2.axhline(2.5e-3, color="gray", lw=0.8, ls=":")
    ax2.text(1015, 2.7e-3, "line width of the figure", fontsize=7,
             color="gray", ha="right", va="bottom")
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("figure5b_validation.png", dpi=200)
    print("wrote figure5b_validation.png")
