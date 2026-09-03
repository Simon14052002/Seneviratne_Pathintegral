"""Plottet die gespeicherten QPU-Zaehlraten direkt als 3 separate PDFs."""
import os
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hardware as hw

LAEUFE = [
    ('ibm_kingston_kry_200fs_mit_reset.npy',   'mit Reset (Mid-Circuit)', 'qpu_200fs_mit_reset'),
    ('ibm_kingston_kry_200fs_ohne_reset.npy',  'aufgeschobene Messung',   'qpu_200fs_ohne_reset'),
    ('ibm_kingston_kry_500fs_ohne_reset.npy',  'aufgeschoben, 500 fs',    'qpu_500fs_ohne_reset'),
]
DT, M, LAM, DELTA, PAIRS = 20.0, 4, 35.0, 10.0, [(0, 1)]
NMIN = 50

os.makedirs('pictures', exist_ok=True)
mod, rho0 = hw.make_model(4, lam=LAM)

for datei, titel, dateiname_praefix in LAEUFE:
    c = list(np.load(datei, allow_pickle=True))
    n_sys = 2
    tmax = max(len(list(d)[0]) for d in c) - n_sys
    kanaele = len(c) // tmax
    floor = (kanaele == 7)
    times = list(range(1, tmax + 1))

    g = hw.build_hardware_grid(n_sites=4, m=M, dt_fs=DT, delta=DELTA,
                               model=mod, rho0=rho0, verbose=False)
    jobs = hw.hardware_circuits(g, times, PAIRS, method='svd',
                                defer=True, floor=floor)
    assert len(jobs) == len(c), f"{datei}: {len(jobs)} Jobs, {len(c)} Zaehlraten"
    rho, err, info = hw.analyze(g, jobs, c, pairs=PAIRS, verbose=False)

    ts, ref, tr = info['t_index'], info['reference'], info['rho_tr']
    T = ts * DT
    t_ex = np.arange(tmax + 1) * DT
    nacc = info['p_success'] * sum(c[0].values())
    gut = nacc >= NMIN
    cols = plt.cm.tab10(np.arange(4))

    # --- Eigene Figure für diese Spalte (2 Zeilen, 1 Spalte) ---
    fig, ax = plt.subplots(2, 1, figsize=(5.0, 6.5), sharex=True)

    # 1. Obere Zeile: Populationen
    for j in range(4):
        ax[0].plot(t_ex, np.real(ref[:, j, j]), color=cols[j], lw=1.6,
                   label=f'Site {j+1}')
        ax[0].errorbar(T[gut], np.real(tr[gut, j, j]), yerr=err[gut, j, j],
                       fmt='s', ms=5, capsize=3, color=cols[j], lw=1.1)
        if (~gut).any():
            ax[0].plot(T[~gut], np.real(tr[~gut, j, j]), 's', ms=5,
                       mfc='white', color=cols[j], alpha=.35)

    ax[0].set_title(f'{titel}\n({len(c)} Kreise, t=1..{tmax})', fontsize=11)
    ax[0].set_ylabel('Population', fontsize=10)
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(ncol=2, fontsize=8, loc='upper right')
    ax[0].grid(alpha=.25)

    # 2. Untere Zeile: Kohärenz
    a, b = PAIRS[0]
    for teil, lab, cc, ee in ((np.real, r'\mathrm{Re}', 'C0', err),
                              (np.imag, r'\mathrm{Im}', 'C2', info['rho_err_im'])):
        ax[1].plot(t_ex, teil(ref[:, a, b]), lw=1.6, color=cc,
                   label=rf'${lab}\,\rho_{{{a+1}{b+1}}}$')
        ax[1].errorbar(T[gut], teil(tr[gut, a, b]), yerr=ee[gut, a, b],
                       fmt='s', ms=5, capsize=3, color=cc, lw=1.1)
        if (~gut).any():
            ax[1].plot(T[~gut], teil(tr[~gut, a, b]), 's', ms=5,
                       mfc='white', color=cc, alpha=.35)

    ax[1].axhline(0, color='0.85', lw=.8, zorder=0)
    ax[1].set_ylabel(rf'$\rho_{{{a+1}{b+1}}}$', fontsize=10)
    ax[1].set_xlabel('t [fs]', fontsize=10)
    ax[1].legend(fontsize=8, loc='upper left')
    ax[1].grid(alpha=.25)

    fig.tight_layout()

    # Als Einzel-PDF speichern
    out_pdf = f'pictures/{dateiname_praefix}.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"  gespeichert: {out_pdf}")