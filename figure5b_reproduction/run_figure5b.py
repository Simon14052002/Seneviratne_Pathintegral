"""
Production run: exact non-Markovian map L(t) for the 4-site FMO pathway
(Seneviratne, Walters & Wang, ACS Omega 2024, Fig. 5) via the path-integral
(TEMPO/TNPI) implementation, followed by the Figures-folder grid pipeline
(L(t) -> Choi -> Kraus -> operator-sum propagation).

Usage:  python3 run_figure5b.py <dt_fs> <t_mem_fs> <eps> <tag>
"""

import sys
import time
import numpy as np

from pathintegral_map import PathIntegralMap
from grid_pipeline import propagate_grid

dt_fs = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
t_mem = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
tag = sys.argv[4] if len(sys.argv) > 4 else "main"

T_TOTAL = 1000.0                      # fs
n_steps = int(round(T_TOTAL / dt_fs))
kmax = int(round(t_mem / dt_fs))

# paper eq. 30 [cm^-1]
H = np.array([[12375.0, -87.7,   5.5,  -5.9],
              [-87.7, 12495.0,  30.8,   8.2],
              [5.5,      30.8, 12175.0, -53.4],
              [-5.9,      8.2, -53.4, 12285.0]])
H = H - np.trace(H) / 4 * np.eye(4)
LAM, GAM, TEMP = 35.0, 106.18, 300.0  # cm^-1, cm^-1, K

print(f"[{tag}] dt = {dt_fs} fs, kmax = {kmax} ({t_mem} fs memory), "
      f"eps = {eps}, {n_steps} steps")
t0 = time.time()
pm = PathIntegralMap(H, dt_fs, kmax, LAM, GAM, TEMP, eps=eps, chi_max=512)


def checkpoint(k, maps_sofar):
    if k % 20 == 0:
        np.savez(f"checkpoint_{tag}.npz", n_done=k, dt_fs=dt_fs,
                 maps=np.array(maps_sofar))


maps = pm.run(n_steps, verbose=True, callback=checkpoint)
print(f"[{tag}] TEMPO done in {time.time() - t0:.1f} s")

# ---- grid pipeline: L(t) -> Choi -> Kraus -> rho(t) ------------------------
rho0 = np.zeros((4, 4), dtype=complex)
rho0[0, 0] = 1.0                       # excitation starts on site 1
rho_dir, rho_kr, n_kraus, min_eig, tr_err = propagate_grid(maps, rho0)

t_list = np.arange(n_steps + 1) * dt_fs
pop_dir = np.real(np.einsum('tii->ti', rho_dir))
pop_kr = np.real(np.einsum('tii->ti', rho_kr))

print(f"[{tag}] max |pop_direct - pop_kraus|    = "
      f"{np.abs(pop_dir - pop_kr).max():.2e}")
print(f"[{tag}] max trace-preservation error    = {tr_err.max():.2e}")
print(f"[{tag}] min Choi eigenvalue             = {min_eig.min():.2e}")
print(f"[{tag}] number of Kraus operators       : "
      f"min {n_kraus.min()}, max {n_kraus.max()}")

np.savez(f"figure5b_{tag}.npz",
         t_list=t_list, maps=np.array(maps), pop_direct=pop_dir,
         pop_kraus=pop_kr, n_kraus=n_kraus, min_choi_eig=min_eig,
         trace_err=tr_err, dt_fs=dt_fs, t_mem_fs=t_mem, eps=eps, H=H,
         lam=LAM, gamma=GAM, T_K=TEMP)
print(f"[{tag}] saved figure5b_{tag}.npz")
