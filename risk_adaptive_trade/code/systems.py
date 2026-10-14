"""Synthetic system builders and thin wrappers around sim_core.simulate."""
import numpy as np
from sim_core import simulate, static_cascade


def torus_d(P, Q):
    d = np.abs(P[:, None, :] - Q[None, :, :])
    d = np.minimum(d, 1 - d)
    return np.sqrt((d ** 2).sum(-1))


def hub_system(K=5, Delta=3.0, xc=0.6, cap_alt=10.0):
    """One importer (demand 1), a low-cost hub (cost 0, restriction threshold xc) and K
    alternatives (cost Delta, ample capacity)."""
    M = K + 1
    N = 1
    A = np.ones((1, N, M))
    C = np.full((1, N, M), Delta)
    C[0, 0, 0] = 0.0
    Dbar = np.ones((1, N))
    Xbar = np.full((1, M), cap_alt)
    Xbar[0, 0] = xc
    slack = np.zeros((1, M))
    Psc = np.zeros((1, M))
    Lch = np.eye(M)[None]
    anom = np.zeros((1, 1, M))
    w0 = np.zeros((1, N, M))
    w0[0, 0, :] = 1.0 / M
    r0 = np.zeros((1, M))
    return dict(A=A, C=C, Dbar=Dbar, Xbar=Xbar, slack=slack, Psc=Psc, Lch=Lch, anom=anom, w0=w0, r0=r0)


def run_hub(beta=2, tau_m=5, tau_r=0.01, b=0.5, q=1.0, p=0.0, z=0.0, T=3000, burn=1000, seed=1, **kw):
    S = hub_system(**kw)
    M = S['A'].shape[2]
    mask = np.zeros((1, M))
    mask[0, 0] = 1
    return simulate(S['A'], S['C'], S['Dbar'], S['Xbar'], S['slack'], S['Psc'], S['Lch'], S['anom'],
                    S['w0'], S['r0'], np.array([float(beta)]), 1.0, tau_m, tau_r, b, q, 0.0, 0.0, p, z,
                    mask, np.zeros(T), T, burn, seed, 0)


def spatial_system(N=100, M=30, slack=0.15, sig_lo=0.05, sig_hi=0.35, ell=0.2, seed=0,
                   headroom=1.5, capsig=0.5, topo='complete', kmean=8):
    """N importers and M exporters placed uniformly on the unit torus.  Trade cost = distance,
    supply shocks Gaussian with correlation exp(-d/ell), exporter-specific volatility sig_j,
    lognormal normal-exportable supply scaled so that total = headroom x total demand."""
    rng = np.random.default_rng(seed)
    PI = rng.random((N, 2))
    PE = rng.random((M, 2))
    C = torus_d(PI, PE)
    dE = torus_d(PE, PE)
    if ell > 0:
        Corr = np.exp(-dE / ell)
    else:
        Corr = np.eye(M)
    Lc = np.linalg.cholesky(Corr + 1e-9 * np.eye(M))
    sig = rng.uniform(sig_lo, sig_hi, M)
    rng2 = np.random.default_rng(seed + 100)
    cap = rng2.lognormal(0, capsig, M)
    cap = cap / cap.sum() * headroom * N
    Acc = np.ones((N, M))
    if topo != 'complete':
        rng3 = np.random.default_rng(seed + 200)
        if topo == 'sparse_er':
            k = np.clip(rng3.poisson(kmean, N), 2, M)
        else:  # heavy-tailed importer degrees (Pareto, exponent ~2.5)
            k = np.clip(np.round(2 + (kmean - 2) * 0.6 * (rng3.pareto(1.5, N) + 1)), 2, M).astype(int)
        # exporter attractiveness heterogeneous (preferential), so exporter degrees are also skewed
        pa = cap / cap.sum()
        Acc = np.zeros((N, M))
        for i in range(N):
            Acc[i, rng3.choice(M, int(k[i]), replace=False, p=pa)] = 1.0
    w0 = Acc / Acc.sum(1, keepdims=True)
    return dict(A=Acc[None], C=C[None], Dbar=np.ones((1, N)), Xbar=cap[None],
                slack=(slack * cap)[None], Psc=(sig * cap)[None], Lch=Lc[None],
                anom=np.zeros((1, 1, M)), w0=w0[None], r0=np.zeros((1, M)),
                Corr=Corr, sig=sig, PE=PE, PI=PI)


def stack_layers(Ss):
    """Stack single-layer systems (same N, M) into a multiplex system."""
    keys = ['A', 'C', 'Dbar', 'Xbar', 'slack', 'Psc', 'Lch', 'anom', 'w0', 'r0']
    return {k: np.concatenate([S[k] for S in Ss], axis=0) for k in keys}


def run_sys(S, beta, gam=6.0, tau_m=5, tau_r=2, b=0.5, q=0.8, kappa=0.0, chi=0.0, p=0.0, z=0.0,
            T=3000, burn=500, seed=1, rho_ext=None, rec=-1):
    L, N, M = S['A'].shape
    rx = np.zeros(T) if rho_ext is None else np.asarray(rho_ext, float)
    bvec = np.full(N, float(beta)) if np.isscalar(beta) else np.asarray(beta, float)
    return simulate(S['A'], S['C'], S['Dbar'], S['Xbar'], S['slack'], S['Psc'], S['Lch'], S['anom'],
                    S['w0'], S['r0'], bvec, gam, tau_m, tau_r, b, q, kappa, chi, p, z,
                    np.zeros((L, M)), rx, T, burn, seed, rec)


def logit_shares(S, l, gam, beta, r):
    U = -gam * S['C'][l] - beta * r[None, :]
    U = np.where(S['A'][l] > 0, U, -np.inf)
    U = U - U.max(1, keepdims=True)
    E = np.exp(U)
    return E / E.sum(1, keepdims=True)


def mf_response(S, l, w, us, b=0.5, q=0.8, ndraw=500, seed=7, anom_rows=None):
    """Noise-averaged static response of layer l with shares w frozen, demand scaled uniformly by u:
    F(u) = E[ load-weighted failure rate ],  Sh(u) = E[ shortfall fraction ],  Phi(u) = E[restricted capacity share].
    This is the load-weighted (degree-weighted) one-dimensional reduction used for the fold analysis."""
    rng = np.random.default_rng(seed)
    N, M = w.shape
    Dbar = S['Dbar'][l]
    Xbar = S['Xbar'][l]
    slack = S['slack'][l]
    Psc = S['Psc'][l]
    loads = (Dbar[:, None] * w).sum(0)
    ell = loads / loads.sum()
    F = np.zeros(len(us))
    Sh = np.zeros(len(us))
    Ph = np.zeros(len(us))
    for d in range(ndraw):
        if anom_rows is not None:
            xi = anom_rows[rng.integers(anom_rows.shape[0])]
        else:
            xi = S['Lch'][l] @ rng.standard_normal(M)
        X = np.maximum(0.0, Xbar + Psc * xi)
        Th = X + slack
        for k, u in enumerate(us):
            o0 = (u * Dbar)[:, None] * w
            o = o0.copy()
            openm = np.zeros(M, dtype=np.bool_)
            deliv = np.zeros((N, M))
            from sim_core import cascade_layer
            nres, req, dj, unmet = cascade_layer(o, w, X, Th, b, q, openm, deliv)
            f = np.where(req > 1e-12, 1 - dj / np.maximum(req, 1e-12), 0.0)
            f = np.clip(f, 0, 1)
            F[k] += (ell * f).sum()
            Sh[k] += 1 - dj.sum() / o0.sum()
            Ph[k] += Xbar[~openm].sum() / Xbar.sum()
    return F / ndraw, Sh / ndraw, Ph / ndraw


def mf_branches(us, F, kappa, tau_m):
    """Stationary branches of rho_{t+1} = lam rho + F(1 + kappa (rho + rho_ext)).
    Parametrise by u: rho = F(u)/(1-lam), rho_ext = (u-1)/kappa - rho.  Returns rho_ext(u), rho(u),
    trigger (local max of rho_ext) and release (subsequent local min), loop gain kappa F'/(1-lam)."""
    lam = np.exp(-1.0 / tau_m)
    rho = F / (1 - lam)
    rext = (us - 1) / kappa - rho
    dF = np.gradient(F, us)
    gain = kappa * dF / (1 - lam)
    trig = np.nan
    rel = np.nan
    unstable = gain > 1
    if unstable.any():
        i0 = np.argmax(unstable)
        i1 = len(us) - 1 - np.argmax(unstable[::-1])
        trig = rext[max(i0 - 1, 0)]
        rel = rext[min(i1 + 1, len(us) - 1)]
    return rext, rho, trig, rel, gain
