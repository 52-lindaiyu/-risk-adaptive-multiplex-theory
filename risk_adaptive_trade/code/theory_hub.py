"""Exact analysis of the single-importer salience map  x_{t+1} = lam x_t + 1[x_t < c]  (x = r/b, c = R_c/b).
Regimes: c <= 0 no restriction; 0 < c < 1/(1-lam) switching (cycles); c >= 1/(1-lam) persistent restriction.
On the invariant interval J = [lam c, lam c + 1) the map is conjugate to the contracted rotation
y -> lam y + delta (mod 1), delta = 1 - lam (1-lam) c, so the restriction frequency omega = 1 - rho(f_delta).
Exact 1/n plateaus: lam^{n-1}/(1-lam^n) < c <= lam^{n-2}/(1-lam^n)  (period exactly n, one restriction per period).
Writes handoff/H_*.csv used by Fig. 2 and Supplementary Note 1."""
import os, sys
import numpy as np
import pandas as pd
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs('handoff', exist_ok=True)


def omega_exact(c, lam, n_trans=4000, n_avg=20000):
    """Restriction frequency of the map from x0 = 0 (low-salience start)."""
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


def regime(c, lam):
    if c <= 0:
        return 'none'
    if c >= 1.0 / (1.0 - lam):
        return 'persistent'
    return 'switching'


def plateau_1n(n, lam):
    return lam ** (n - 1) / (1 - lam ** n), lam ** (n - 2) / (1 - lam ** n)


if __name__ == '__main__':
    from numba import njit

    @njit(cache=False)
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

    # (1) staircase and exact 1/n plateaus for tau_m = 5 and 10
    rows = []
    for tm in (5.0, 10.0):
        lam = np.exp(-1 / tm)
        cs = np.geomspace(0.01, 1.2 / (1 - lam), 4000)
        for c in cs:
            w = om(c, lam, 5000, 40000)
            rows.append(dict(tau_m=tm, c=c, omega=w, regime=regime(c, lam)))
    St = pd.DataFrame(rows)
    St.to_csv('handoff/H_staircase.csv', index=False)
    # verify plateau formula: every c inside an exact 1/n interval has omega == 1/n
    chk = []
    for tm in (5.0, 10.0):
        lam = np.exp(-1 / tm)
        d = St[St.tau_m == tm]
        for n in range(2, 60):
            lo, hi = plateau_1n(n, lam)
            sel = d[(d.c > lo) & (d.c <= hi)]
            if len(sel):
                chk.append(dict(tau_m=tm, n=n, n_pts=len(sel), max_dev=float(np.abs(sel.omega - 1 / n).max())))
    pd.DataFrame(chk).to_csv('handoff/H_plateau_check.csv', index=False)
    # (2) completeness: share of switching-regime c whose omega is a rational with denominator <= Q
    comp = []
    for tm in (5.0, 10.0):
        lam = np.exp(-1 / tm)
        d = St[(St.tau_m == tm) & (St.regime == 'switching')]
        # uniform-in-c resampling for a Lebesgue-measure estimate
        cs = np.linspace(1e-3, 1 / (1 - lam) - 1e-3, 6000)
        oms = np.array([om(c, lam, 5000, 60000) for c in cs])
        for Q in (5, 10, 20, 40, 80):
            fr = [Fraction(int(round(o * 60000)), 60000).limit_denominator(Q) for o in oms]
            ok = np.array([abs(float(f) - o) < 2e-5 for f, o in zip(fr, oms)])
            comp.append(dict(tau_m=tm, Q=Q, share_rational=ok.mean()))
    pd.DataFrame(comp).to_csv('handoff/H_completeness.csv', index=False)
    # (3) period bounds: tau_m ln(1+1/(lam c)) <= 1/omega <= tau_m ln(1+1/(lam^2 c)) in the switching regime
    rows = []
    for tm in (2.0, 3.0, 5.0, 10.0, 20.0):
        lam = np.exp(-1 / tm)
        for c in np.geomspace(0.005, 0.999 / (1 - lam), 800):
            w = om(c, lam, 5000, 40000)
            if w <= 0 or w >= 1:
                continue
            T = 1 / w
            rows.append(dict(tau_m=tm, c=c, T=T, lo=tm * np.log(1 + 1 / (lam * c)), hi=tm * np.log(1 + 1 / (lam ** 2 * c)),
                             cont=tm * np.log(1 + 1 / c)))
    B = pd.DataFrame(rows)
    B['inside'] = (B['T'] >= B['lo'] - 1e-9) & (B['T'] <= B['hi'] + 1e-9)
    B.to_csv('handoff/H_bounds.csv', index=False)
    # (4) regime map in (beta, tau_m) for the hub of Fig. 2 (K = 0.985, b = 0.5)
    K = 3 - np.log(5 * 0.6 / 0.4)
    rows = []
    for tm in np.geomspace(1, 50, 60):
        lam = np.exp(-1 / tm)
        for beta in np.geomspace(0.05, 30, 80):
            c = K / (beta * 0.5)
            w = om(c, lam, 5000, 20000)
            rows.append(dict(tau_m=tm, beta=beta, c=c, omega=w, regime=regime(c, lam)))
    pd.DataFrame(rows).to_csv('handoff/H_regimes.csv', index=False)
    print('done')
