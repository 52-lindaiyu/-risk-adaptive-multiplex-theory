"""Wrappers for the generalised simulator (sim_core2)."""
import numpy as np
from sim_core2 import simulate2, static_scan2, expected_failure
from systems import spatial_system, hub_system, stack_layers, torus_d


def run2(S, beta, tau_m=5.0, tau_r=2.0, gam=6.0, b=0.5, q=0.8, mode=0, sw=0.05, kappa=0.0, kmax=0.0,
         chi=None, cap=0.0, mu=1.0, noise_sd=0.0, p=0.0, z=0.0, shock_mode=0, shock_mask=None,
         T=3000, burn=500, seed=1, rho_ext=None, rec=-1, tau_m_i=None, tau_r_i=None):
    """beta, tau_m_i, tau_r_i may be scalars or length-N arrays (importer heterogeneity).
    tau_m sets the public-salience memory and, unless tau_m_i is given, every importer's private memory."""
    L, N, M = S['A'].shape
    bvec = np.full(N, float(beta)) if np.isscalar(beta) else np.asarray(beta, float)
    tmi = np.full(N, float(tau_m)) if tau_m_i is None else np.asarray(tau_m_i, float)
    tri = np.full(N, float(tau_r)) if tau_r_i is None else np.asarray(tau_r_i, float)
    lam_i = np.exp(-1.0 / tmi)
    eta_i = 1.0 - np.exp(-1.0 / tri)
    rx = np.zeros(T) if rho_ext is None else np.asarray(rho_ext, float)
    chi = np.zeros((L, L)) if chi is None else np.asarray(chi, float)
    sm = np.zeros((L, M)) if shock_mask is None else np.asarray(shock_mask, float)
    return simulate2(S['A'], S['C'], S['Dbar'], S['Xbar'], S['slack'], S['Psc'], S['Lch'], S['anom'], S['w0'],
                     bvec, lam_i, eta_i, float(np.exp(-1.0 / tau_m)), float(mu), float(noise_sd),
                     float(gam), float(b), float(q), int(mode), float(sw), float(kappa), float(kmax), chi, float(cap),
                     float(p), float(z), int(shock_mode), sm, rx, int(T), int(burn), int(seed), int(rec))


def run_hub2(beta=2.0, tau_m=5.0, tau_r=0.01, b=0.5, q=1.0, p=0.0, z=0.0, shock_mode=0, T=3000, burn=1000,
             seed=1, **kw):
    S = hub_system(**kw)
    M = S['A'].shape[2]
    mask = np.zeros((1, M))
    mask[0, 0] = 1
    return run2(S, beta, tau_m=tau_m, tau_r=tau_r, gam=1.0, b=b, q=q, p=p, z=z, shock_mode=shock_mode,
                shock_mask=mask, T=T, burn=burn, seed=seed, rec=0)


def mf_response2(S, l, w, us, b=0.5, q=0.8, mode=0, sw=0.05, ndraw=500, seed=7, anom_rows=None):
    """F(u), order shortfall, baseline shortfall and restricted share from the static scan (frozen shares)."""
    N, M = w.shape
    if anom_rows is None:
        rng = np.random.default_rng(seed)
        anom_rows = (S['Lch'][l] @ rng.standard_normal((M, 2000))).T.copy()
    Sh, Sb, Ph, Fl = static_scan2(S['Dbar'][l], np.ascontiguousarray(w), S['Xbar'][l], S['Psc'][l], S['slack'][l],
                                  np.ascontiguousarray(anom_rows), np.asarray(us, float), ndraw, seed, b, q, mode, sw)
    return Fl.mean(0), Sh.mean(0), Sb.mean(0), Ph.mean(0), Fl


def mf_branches_exact(us, F, kappa, tau_m):
    """Stationary branches of rho' = lam rho + F(1+kappa(rho+rho_ext)) with the exact discrete gain
    G = kappa F'(u)/(1 - exp(-1/tau_m)).  Returns rext(u), rho(u), gain(u), list of folds (u, rext, type)."""
    lam = np.exp(-1.0 / tau_m)
    rho = F / (1 - lam)
    rext = (us - 1) / kappa - rho
    dF = np.gradient(F, us)
    gain = kappa * dF / (1 - lam)
    folds = []
    for k in range(1, len(us)):
        if gain[k - 1] <= 1 < gain[k]:
            folds.append((us[k], rext[k], 'collapse'))
        if gain[k - 1] > 1 >= gain[k]:
            folds.append((us[k], rext[k], 'recovery'))
    return rext, rho, gain, folds


def classify_regime(us, F, kappa, tau_m):
    """0 monostable, 1 reversible hysteresis (recovery point >= 0), 2 persistent (recovery < 0 <= collapse),
    3 pre-fragmented (collapse point < 0); also returns the outermost collapse/recovery points."""
    rext, rho, gain, folds = mf_branches_exact(us, F, kappa, tau_m)
    col = [f[1] for f in folds if f[2] == 'collapse']
    rec = [f[1] for f in folds if f[2] == 'recovery']
    if not col:
        return 0, np.nan, np.nan
    k0 = int(np.argmax(gain > 1))
    trig = rext[:k0].max() if k0 > 0 else -np.inf
    i1 = len(us) - 1 - np.argmax((gain > 1)[::-1])
    rel = rext[i1:].min() if i1 < len(us) else np.nan
    if trig < 0:
        return 3, trig, rel
    if rel < 0:
        return 2, trig, rel
    return 1, trig, rel
