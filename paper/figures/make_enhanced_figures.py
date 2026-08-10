"""
Publication figures for the memory-window (transfer-tensor) enhanced algorithm.
Regenerated from the saved dynamical maps (maps_to_grid/normal/data) via the
pure-numpy grid reference in grid_ref.py, which reproduces the reused Qiskit
circuit exactly. Three maps: Lindblad (semigroup, variant 1), HEOM and path
integral (non-Markovian, variant 2). 4-site FMO, dt = 10 fs, t up to 1000 fs.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import grid_ref as g

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "maps_to_grid", "normal", "data")

mpl.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "font.size": 9,
    "axes.labelsize": 9.5, "legend.fontsize": 7.4, "legend.handlelength": 1.5,
    "legend.labelspacing": 0.3, "legend.columnspacing": 1.1,
    "legend.handletextpad": 0.5, "xtick.labelsize": 8.3, "ytick.labelsize": 8.3,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.2, "ytick.major.size": 3.2,
    "axes.unicode_minus": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL, DCOL = 3.375, 7.0
COLOR = {"pi": "#d62728", "heom": "#1f77b4", "lind": "#2ca02c"}
DT = 10.0
K_PI, K_HEOM = 15, 25


def load():
    lind = g.load_maps(os.path.join(DATA, "lindblad_maps.npz"))
    heom = g.load_maps(os.path.join(DATA, "heom_maps.npz"))
    pi = g.load_maps(os.path.join(DATA, "pathintegral_maps.npz"))
    return lind, heom, pi


def compute():
    lind, heom, pi = load()
    t = lind["t_fs"]
    d = 4
    rho0 = np.zeros((d, d), complex)
    rho0[0, 0] = 1.0
    n = len(t) - 1
    bench = {"lind": g.pops_direct(lind["maps"], rho0),
             "heom": g.pops_direct(heom["maps"], rho0)}
    bench["pi"] = bench["heom"]                        # exact ref for PI
    grid = {}
    grid["lind"] = g.pops_enhanced_markov(lind["maps"][1], d, rho0, n)
    grid["heom"], Tn_h = g.pops_enhanced_memory(heom["maps"], rho0, K_HEOM, n)
    grid["pi"], Tn_p = g.pops_enhanced_memory(pi["maps"], rho0, K_PI, n)
    k1 = g.pops_enhanced_memory(pi["maps"], rho0, 1, n)[0]
    Tn_l = np.array([np.linalg.norm(t, 2)
                     for t in g.transfer_tensors(lind["maps"], K_HEOM)])
    Tn = {"pi": Tn_p, "heom": Tn_h, "lind": Tn_l}
    # K sweep
    Ks = [1, 5, 10, 15, 20, 25]
    sweep = {"pi": [], "heom": []}
    for K in Ks:
        pP = g.pops_enhanced_memory(pi["maps"], rho0, K, n)[0]
        pH = g.pops_enhanced_memory(heom["maps"], rho0, K, n)[0]
        sweep["pi"].append(np.abs(pP - bench["heom"]).max())
        sweep["heom"].append(np.abs(pH - bench["heom"]).max())
    return dict(t=t, bench=bench, grid=grid, k1=k1, Tn=Tn,
                Ks=np.array(Ks), sweep=sweep)


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", name)


# ==========================================================================
# FIG grid: 3 maps on one reused circuit vs exact solver, memory-window marks
# ==========================================================================
def fig_grid(R):
    t = R["t"]
    names = [("pi", "path integral"), ("heom", "HEOM"), ("lind", "Lindblad")]
    fig, axes = plt.subplots(2, 2, figsize=(COL, 3.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for s, ax in enumerate(axes):
        for key, _ in names:
            ax.plot(t, R["grid"][key][:, s], color=COLOR[key], lw=1.5)
            ax.plot(t, R["bench"][key][:, s], color="k", ls=(0, (4, 2)), lw=0.9)
        ax.axvline(K_PI * DT, color=COLOR["pi"], lw=0.8, ls=":", alpha=0.8)
        ax.axvline(K_HEOM * DT, color=COLOR["heom"], lw=0.8, ls=":", alpha=0.8)
        ax.set_xlim(0, 1000)
        ax.set_ylim(-0.03, 1.03)
        ax.text(0.94, 0.9, f"site {s+1}", transform=ax.transAxes, ha="right",
                va="top", fontsize=8)
        ax.tick_params(labelsize=7.6)
    for ax in (axes[2], axes[3]):
        ax.set_xlabel("time  [fs]", fontsize=8.6)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Population", fontsize=8.6)
    handles = [Line2D([], [], color=COLOR[k], lw=1.6, label=lab)
               for k, lab in names]
    handles.append(Line2D([], [], color="k", ls=(0, (4, 2)), lw=1.0,
                          label="exact solver"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.54, 1.06), columnspacing=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_grid.pdf")


# ==========================================================================
# FIG memory: transfer-tensor decay + accuracy vs memory window
# ==========================================================================
def fig_memory(R):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(COL, 1.95))
    for key, lab in [("lind", "Lindblad"), ("heom", "HEOM"),
                     ("pi", "path int.")]:
        Tn = np.maximum(R["Tn"][key], 1e-16)
        a0.semilogy(np.arange(1, len(Tn) + 1) * DT, Tn, "o-", ms=2.8,
                    color=COLOR[key], lw=1.1, label=lab)
    a0.set_xlabel(r"$m\,\Delta t$  [fs]", fontsize=8.4)
    a0.set_ylabel(r"$\|T_m\|_2$", fontsize=8.6)
    a0.set_ylim(1e-17, 3e0)
    a0.grid(True, which="major", ls=":", lw=0.5, alpha=0.5)
    a0.legend(frameon=False, fontsize=6.6, loc="lower left", ncol=1)
    a0.tick_params(labelsize=7.4)
    for key, lab in [("pi", "path int."), ("heom", "HEOM")]:
        a1.semilogy(R["Ks"] * DT, R["sweep"][key], "o-", ms=3.5,
                    color=COLOR[key], lw=1.2, label=lab)
    a1.set_xlabel(r"memory window $K\,\Delta t$  [fs]", fontsize=8.4)
    a1.set_ylabel(r"$\max_i|\Delta P_i|$", fontsize=8.6)
    a1.grid(True, which="major", ls=":", lw=0.5, alpha=0.5)
    a1.legend(frameon=False, fontsize=6.8, loc="upper right")
    a1.tick_params(labelsize=7.4)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_memory.pdf")


# ==========================================================================
# FIG deviation: quantum-grid vs exact solver, incl. K=1 memory-discarded fail
# ==========================================================================
def fig_deviation(R):
    t = R["t"]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    floor = 1e-16
    ax.semilogy(t, np.maximum(np.abs(R["k1"] - R["bench"]["pi"]).max(1), floor),
                color="#8a8a8a", lw=1.5,
                label="path integral, memory discarded ($K=1$)")
    for key, lab in [("pi", "path integral"), ("heom", "HEOM"),
                     ("lind", "Lindblad")]:
        dev = np.maximum(np.abs(R["grid"][key] - R["bench"][key]).max(1), floor)
        ax.semilogy(t, dev, color=COLOR[key], lw=1.5, label=lab)
    ax.set_xlim(0, 1000)
    ax.set_ylim(1e-16, 3e0)
    ax.set_xlabel("time  [fs]")
    ax.set_ylabel(r"$\max_i|P_i^{\mathrm{grid}}-P_i^{\mathrm{exact}}|$",
                  fontsize=8.6)
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=2,
              frameon=False, fontsize=7.0)
    save(fig, "fig_deviation.pdf")


# ==========================================================================
# FIG validity: companion-operator spectrum (stability) + physicality
# ==========================================================================
def fig_validity():
    lind, heom, pi = load()
    d = 4
    rho0 = np.zeros((d, d), complex)
    rho0[0, 0] = 1.0
    n = len(lind["t_fs"]) - 1
    t = lind["t_fs"]
    rho_h, E_h = g.traj_and_E_memory(heom["maps"], rho0, K_HEOM, n)
    rho_p, E_p = g.traj_and_E_memory(pi["maps"], rho0, K_PI, n)
    rho_l = g.traj_markov(lind["maps"][1], d, rho0, n)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(COL, 1.95))
    # (a) spectrum of the companion operator in the unit disk
    th = np.linspace(0, 2 * np.pi, 200)
    a0.plot(np.cos(th), np.sin(th), color="#555", lw=0.8)
    for key, E in [("heom", E_h), ("pi", E_p)]:
        ev = np.linalg.eigvals(E)
        a0.plot(ev.real, ev.imag, "o", ms=2.4, mfc="none", mew=0.7,
                color=COLOR[key], label={"heom": "HEOM", "pi": "path int."}[key])
    a0.set_aspect("equal")
    a0.set_xlim(-1.25, 1.25)
    a0.set_ylim(-1.25, 1.25)
    a0.set_xticks([-1, 0, 1])
    a0.set_yticks([-1, 0, 1])
    a0.set_xlabel(r"$\mathrm{Re}\,\lambda(E)$", fontsize=8.4)
    a0.set_ylabel(r"$\mathrm{Im}\,\lambda(E)$", fontsize=8.4)
    a0.tick_params(labelsize=7.4)
    a0.legend(frameon=False, fontsize=6.6, loc="upper left",
              handletextpad=0.2, borderpad=0.1)
    # (b) trace preservation over time
    for key, rho in [("pi", rho_p), ("heom", rho_h), ("lind", rho_l)]:
        tr = np.abs(np.trace(rho, axis1=1, axis2=2) - 1.0)
        a1.semilogy(t, np.maximum(tr, 1e-16), color=COLOR[key], lw=1.3,
                    label={"pi": "path int.", "heom": "HEOM",
                           "lind": "Lindblad"}[key])
    a1.set_xlim(0, 1000)
    a1.set_ylim(1e-16, 1e-3)
    a1.set_xlabel("time  [fs]", fontsize=8.4)
    a1.set_ylabel(r"$|\mathrm{Tr}\,\rho-1|$", fontsize=8.4)
    a1.grid(True, which="major", ls=":", lw=0.5, alpha=0.5)
    a1.legend(frameon=False, fontsize=6.6, loc="lower right", ncol=1)
    a1.tick_params(labelsize=7.4)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_validity.pdf")


if __name__ == "__main__":
    R = compute()
    fig_grid(R)
    fig_memory(R)
    fig_deviation(R)
    fig_validity()
    print("done")
