"""Checks of the exact hub map against the full simulator: restriction frequency, finite friction,
ratchet boundary with salience impulses, and long-average recheck of apparent bound violations.
Outputs handoff/H_sim_vs_exact.csv, H_finite_taur.csv, H_ratchet_impulse.csv, H_bounds_recheck.csv."""
import os, sys
import numpy as np
import pandas as pd
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from systems2 import run_hub2

K = 3 - np.log(5 * 0.6 / 0.4)
KH = 3 + np.log((1 - 0.4) / (5 * 0.4))


@njit
def om(c, lam, n_trans, n_avg):
    x = 0.0
    for _ in range(n_trans):
        x = lam * x + (1.0 if x < c else 0.0)
    k = 0
    for _ in range(n_avg):
        if x < c:
            k += 1
            x = lam * x + 1.0
        else:
            x = lam * x
    return k / n_avg


if __name__ == '__main__':
    rows = []
    for tm in (2.0, 5.0, 10.0):
        lam = np.exp(-1 / tm)
        for beta in np.geomspace(0.3, 60, 25):
            c = K / (0.5 * beta)
            o = run_hub2(beta=beta, tau_m=tm, tau_r=0.01, T=12000, burn=2000)
            rows.append((tm, beta, c, (o[3][:, 0] > 0).mean(), om(c, lam, 20000, 400000)))
    pd.DataFrame(rows, columns=['tau_m', 'beta', 'c', 'omega_sim', 'omega_exact']).to_csv('handoff/H_sim_vs_exact.csv', index=False)
    rows = []
    for tm in (2.0, 5.0, 10.0):
        lam = np.exp(-1 / tm)
        for beta in (2.0, 5.0, 10.0, 20.0, 40.0):
            for tr in (0.01, 0.5, 1.0, 2.0, 3.0, 5.0):
                o = run_hub2(beta=beta, tau_m=tm, tau_r=tr, T=20000, burn=4000)
                r = (o[3][:, 0] > 0).astype(int)
                on = np.sum(np.diff(r) == 1)
                c = K / (0.5 * beta)
                rows.append((tm, beta, tr, c, len(r) / on if on > 0 else np.inf, 1 / om(c, lam, 20000, 400000)))
    pd.DataFrame(rows, columns=['tau_m', 'beta', 'tau_r', 'c', 'T_sim', 'T_exact0']).to_csv('handoff/H_finite_taur.csv', index=False)
    rows = []
    for ts in (3.0, 5.0, 10.0, 20.0):
        for tm in np.geomspace(0.3, 40, 18):
            for beta in (1.0, 3.0, 10.0):
                o = run_hub2(beta=beta, tau_m=tm, tau_r=0.01, b=1.0, q=1.0, p=1 / ts, z=1.0, shock_mode=1, T=40000,
                             burn=2000, xc=0.9)
                x = o[9][:, 0]
                rows.append((ts, tm, beta, (x > 0.4).mean(), tm / ts, 1 / np.log(1 + beta / KH)))
    pd.DataFrame(rows, columns=['tau_s', 'tau_m', 'beta', 'frac_conc', 'ratio', 'boundary']).to_csv(
        'handoff/H_ratchet_impulse.csv', index=False)
    Bd = pd.read_csv('handoff/H_bounds.csv')
    rows = []
    for r in Bd[~Bd.inside].itertuples():
        lam = np.exp(-1 / r.tau_m)
        T = 1 / om(r.c, lam, 20000, 4000000)
        rows.append(dict(tau_m=r.tau_m, c=r.c, T_precise=T, lo=r.lo, hi=r.hi,
                         inside=(T >= r.lo - 1e-9) & (T <= r.hi + 1e-9)))
    pd.DataFrame(rows).to_csv('handoff/H_bounds_recheck.csv', index=False)
