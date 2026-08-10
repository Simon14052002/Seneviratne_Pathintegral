"""Validation battery for the path-integral map implementation."""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad

from pathintegral_map import (PathIntegralMap, brute_force_maps,
                              drude_expansion, bath_correlation,
                              eta_coefficients, FS_TO_CM, KB_CM)
from grid_pipeline import choi_from_map, kraus_from_choi, vec, unvec

# 4-site FMO pathway Hamiltonian, paper eq. 30 [cm^-1]
H_FMO = np.array([[12375.0, -87.7,   5.5,  -5.9],
                  [-87.7, 12495.0,  30.8,   8.2],
                  [5.5,      30.8, 12175.0, -53.4],
                  [-5.9,      8.2, -53.4, 12285.0]])
H_FMO = H_FMO - np.trace(H_FMO) / 4 * np.eye(4)   # remove irrelevant shift
LAM, GAM, TEMP = 35.0, 106.18, 300.0

ok = True


def check(name, err, tol):
    global ok
    stat = "PASS" if err < tol else "FAIL"
    if err >= tol:
        ok = False
    print(f"  [{stat}] {name}:  err = {err:.3e}  (tol {tol:.0e})")


print("1) Bath correlation function: Matsubara expansion vs quadrature")
beta = 1.0 / (KB_CM * TEMP)
a, nu = drude_expansion(LAM, GAM, beta, 200000)


def C_quad(tau):
    """Independent reference for C(tau):
    Im part analytic  (-lam*gam*e^{-gam*tau});  Re part by cosine-weight
    quadrature on [0, W] (coth ~ 1 beyond) plus analytic Ci-function tail."""
    from scipy.special import sici

    def J(w):
        return 2 * LAM * GAM * w / (w ** 2 + GAM ** 2)

    def re_igrand(w):
        if w == 0.0:
            return 4 * LAM / (beta * GAM) / np.pi   # w->0 limit of J/tanh
        return J(w) / np.pi / np.tanh(beta * w / 2)

    W = 2e4   # cm^-1: coth(beta w/2) - 1 < 1e-40 there
    re = quad(re_igrand, 0, W, weight='cos', wvar=tau, limit=2000)[0]
    re += -(2 * LAM * GAM / np.pi) * sici(W * tau)[1]   # tail: J ~ 2 lam gam / w
    im = -LAM * GAM * np.exp(-GAM * tau)
    return re + 1j * im


for t_fs in [2.0, 10.0, 50.0, 150.0]:
    tau = t_fs * FS_TO_CM
    c_m = bath_correlation(tau, a, nu)[0]
    c_q = C_quad(tau)
    check(f"C(tau) at {t_fs:5.1f} fs", abs(c_m - c_q) / abs(c_q), 1e-6)

print("2) eta coefficients: analytic (Matsubara) vs independent quadrature")
dt_fs = 10.0
dtau = dt_fs * FS_TO_CM
eta = eta_coefficients(a, nu, dtau, 4)
xg, wg = np.polynomial.legendre.leggauss(48)


def gauss(f, lo, hi):
    x = 0.5 * (hi - lo) * (xg + 1) + lo
    return 0.5 * (hi - lo) * np.sum(wg * np.array([f(xi) for xi in x]))


# eta[0] = int_0^dt (dt-u) C(u) du   (C varies fast near u=0 -> split)
val0 = sum(gauss(lambda ui: (dtau - ui) * C_quad(ui), a_, b_)
           for a_, b_ in [(0, dtau / 512), (dtau / 512, dtau / 64),
                          (dtau / 64, dtau / 8), (dtau / 8, dtau)])
check("eta[0]", abs(val0 - eta[0]) / abs(eta[0]), 1e-5)
# eta[k] = int_{-dt}^{dt} (dt-|v|) C(k dt + v) dv;  kink at v=0 -> split
for k in [1, 3]:
    valk = (gauss(lambda vi: (dtau + vi) * C_quad(k * dtau + vi), -dtau, 0)
            + gauss(lambda vi: (dtau - vi) * C_quad(k * dtau + vi), 0, dtau))
    check(f"eta[{k}]", abs(valk - eta[k]) / abs(eta[k]), 1e-5)

print("3) Zero coupling limit reproduces unitary dynamics")
pm = PathIntegralMap(H_FMO, dt_fs, kmax=5, lam_cm=0.0, gamma_cm=GAM,
                     T_K=TEMP, eps=0.0, chi_max=1024)
maps = pm.run(6, verbose=False)
err = 0.0
for k, L in enumerate(maps):
    U = expm(-1j * H_FMO * k * dtau)
    err = max(err, np.abs(L - np.kron(U.conj(), U)).max())
check("max |L(t) - Ubar(x)U|", err, 1e-10)

print("4) MPS contraction vs brute-force path sum (full memory)")
bf = brute_force_maps(H_FMO, dt_fs, 4, kmax=6, lam_cm=LAM, gamma_cm=GAM,
                      T_K=TEMP)
pm = PathIntegralMap(H_FMO, dt_fs, kmax=6, lam_cm=LAM, gamma_cm=GAM,
                     T_K=TEMP, eps=0.0, chi_max=4096)
maps = pm.run(4, verbose=False)
err = max(np.abs(maps[k] - bf[k]).max() for k in range(5))
check("max |L_mps - L_bruteforce|", err, 1e-12)

print("5) MPS vs brute force with memory truncation (kmax=2 < n_steps)")
bf = brute_force_maps(H_FMO, dt_fs, 4, kmax=2, lam_cm=LAM, gamma_cm=GAM,
                      T_K=TEMP)
pm = PathIntegralMap(H_FMO, dt_fs, kmax=2, lam_cm=LAM, gamma_cm=GAM,
                     T_K=TEMP, eps=0.0, chi_max=4096)
maps = pm.run(4, verbose=False)
err = max(np.abs(maps[k] - bf[k]).max() for k in range(5))
check("max |L_mps - L_bruteforce|", err, 1e-12)

print("6) Choi -> Kraus roundtrip on a known channel")
rng = np.random.default_rng(7)
A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
K1 = expm(-1j * (A + A.conj().T))          # unitary
K1 = K1 @ np.diag([1.0, 0.8, 0.6, 0.4])    # non-unitary part
K2 = np.zeros((4, 4), dtype=complex)
K2[0, 1], K2[1, 2], K2[2, 3] = 0.6, 0.8, 0.9166
# normalize to CPTP: sum K^dag K = I  (construct completion)
S = K1.conj().T @ K1 + K2.conj().T @ K2
from scipy.linalg import sqrtm
K3 = sqrtm(np.eye(4) - S * 0.5).astype(complex)
K1 *= np.sqrt(0.5)
K2 *= np.sqrt(0.5)
Ltest = sum(np.kron(K.conj(), K) for K in (K1, K2, K3))
C = choi_from_map(Ltest, 4)
kraus, mineig = kraus_from_choi(C, 4)
rho = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
rho = rho @ rho.conj().T
rho /= np.trace(rho)
r1 = unvec(Ltest @ vec(rho), 4)
r2 = sum(M @ rho @ M.conj().T for M in kraus)
check("|eps(rho) via L - via Kraus|", np.abs(r1 - r2).max(), 1e-10)

print()
print("ALL PASS" if ok else "SOME CHECKS FAILED")
