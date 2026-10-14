"""Revision experiments (synthetic).  Usage: python3 exp_rev.py <nproc> [blocks]
Blocks:  P  diversification paradox, robustness variants            -> out_rev/P_paradox.csv
         B  unilateral best responses (single focal importer, CRN)   -> out_rev/B_bestresponse.csv
         F  fold: scalar closure vs simulation on 10 networks        -> out_rev/F_regimes.csv, F_mf.npz
         C  fold: convergence (ramp speed, draws, grid)              -> out_rev/C_ramp.csv, C_mf.csv
         H  fold: heterogeneous deterministic map vs scalar closure  -> out_rev/H_hetmap.csv
         R  residence times of the fragmented state vs system size   -> out_rev/R_residence.csv
         A  alternative restriction / demand closures               -> out_rev/A_closures.csv
         S  multiplex substitution of unserved baseline demand       -> out_rev/S_subst.csv
         E  early-warning ramps + innovation-variance diagnosis      -> out_rev/E_ews.npz
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from systems import spatial_system, stack_layers
from systems2 import run2, mf_response2, mf_branches_exact, classify_regime
from sim_core2 import expected_failure

OUT = 'out_rev'
os.makedirs(OUT, exist_ok=True)
GAM = 6.0
BETAS = [0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 120]
US = np.linspace(0.6, 4.0, 171)
REG = {0: 'mono', 1: 'hyst', 2: 'trap', 3: 'prefrag'}


def cvar(x, a=0.95):
    q = np.quantile(x, a)
    return x[x >= q].mean()


# ------------------------------------------------------------------ P: paradox
VARIANTS = {
    'base': ({}, {}),
    'mixed_mu0.5': ({}, dict(mu=0.5)),
    'private': ({}, dict(mu=0.0)),
    'noisy_public': ({}, dict(noise_sd=0.5)),
    'noisy_private': ({}, dict(mu=0.0, noise_sd=0.5)),
    'het_public': ({}, dict(het=True, mu=1.0)),
    'het_private': ({}, dict(het=True, mu=0.0)),
    'smooth': ({}, dict(mode=1, sw=0.05)),
    'rationing': ({}, dict(mode=2)),
    'headroom1.2': (dict(headroom=1.2), {}),
    'headroom2.0': (dict(headroom=2.0), {}),
    'tau_m2': ({}, dict(tau_m=2.0)),
    'tau_m10': ({}, dict(tau_m=10.0)),
    'no_memory': ({}, dict(tau_m=0.05)),
    'iid_shocks': (dict(ell=0.0), {}),
    'tau_r5': ({}, dict(tau_r=5.0)),
}


def het_params(net, N):
    rng = np.random.default_rng(10000 + net)
    tmi = rng.uniform(1.0, 9.0, N)
    tri = rng.uniform(0.5, 3.5, N)
    bm = rng.lognormal(0.0, 0.5, N)
    return tmi, tri, bm / bm.mean()


def p_task(args):
    var, net, beta = args
    skw, rkw = VARIANTS[var]
    rkw = dict(rkw)
    S = spatial_system(seed=net, **skw)
    N = S['Dbar'].shape[1]
    b = float(beta)
    if rkw.pop('het', False):
        tmi, tri, bm = het_params(net, N)
        b = beta * bm
        rkw['tau_m_i'] = tmi
        rkw['tau_r_i'] = tri
    rkw.setdefault('tau_m', 5.0)
    rkw.setdefault('tau_r', 2.0)
    o = run2(S, b, gam=GAM, T=8000, burn=1000, seed=1000 + net, **rkw)
    so, sb, nr = o[0][:, 0], o[1][:, 0], o[3][:, 0]
    return dict(variant=var, net=net, beta=beta, short_base=sb.mean(), short_order=so.mean(),
                short_base_cvar95=cvar(sb), short_base_q99=np.quantile(sb, 0.99), p_sys5=(nr >= 5).mean(),
                hhi_agg=o[4][:, 0].mean(), hhi_imp=o[5][:, 0].mean(), neff=(1.0 / o[5][:, 0]).mean(),
                salience=o[6][:, 0].mean(), port_corr=o[7][:, 0].mean(), imp_short_sd=o[8].mean(0).std(),
                total_demand=float(S['Dbar'].sum()))


# ------------------------------------------------------------------ B: unilateral best responses
BPOP = [0, 1, 2, 3, 4, 6, 10, 20]
BI = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 60]


def b_task(args):
    mu, tau_m, net, k, bpop = args
    S = spatial_system(seed=net)
    N = S['Dbar'].shape[1]
    focal = int(np.random.default_rng(500 + net).choice(N, 2, replace=False)[k])
    seed = 20000 + 1000 * net + 100 * k + int(10 * bpop) + int(tau_m)
    out = []
    for bi in BI:
        bv = np.full(N, float(bpop))
        bv[focal] = bi
        o = run2(S, bv, gam=GAM, tau_m=tau_m, tau_r=2.0, mu=mu, T=8000, burn=1000, seed=seed)
        shi = o[8][:, focal]
        out.append(dict(mu=mu, tau_m=tau_m, net=net, focal=focal, beta_pop=bpop, beta_i=bi,
                        focal_short=shi.mean(), focal_cvar95=cvar(shi), sys_short=o[1][:, 0].mean()))
    return out


# ------------------------------------------------------------------ fold helpers
def base_state(net, beta=2.0, N=100, M=30, slack=0.4):
    S = spatial_system(seed=net, slack=slack, N=N, M=M)
    o = run2(S, beta, gam=GAM, tau_m=5, tau_r=2, T=3000, burn=2999, seed=99)
    S['w0'] = o[11].copy()
    return S


def detect(rext, phi, T2, thr=0.5, k=200):
    ph = np.convolve(phi, np.ones(k) / k, 'same')
    up, dn = ph[:T2], ph[T2:]
    xs_up, xs_dn = rext[:T2], rext[T2:]
    if up[k:2 * k].mean() > thr:
        return 'prefrag', np.nan, np.nan
    iu = np.where(up > thr)[0]
    trig = xs_up[iu[0]] if len(iu) else np.nan
    if np.isnan(trig):
        return 'mono', np.nan, np.nan
    above = np.where(dn > thr)[0]
    if len(above) == 0:
        return 'hyst', trig, xs_dn[0]
    last = above[-1]
    if last >= len(dn) - k:
        return 'trap', trig, np.nan
    return 'hyst', trig, xs_dn[last]


MODES = {'frozen': (0.0, 1e9), 'adaptive': (2.0, 2.0)}
_MF = {}


def mf_curve(net, ndraw=600, us=US, N=100, M=30, mode=0, sw=0.05):
    key = (net, ndraw, len(us), N, M, mode, sw)
    if key not in _MF:
        S = base_state(net, N=N, M=M)
        F, Sh, Sb, Ph, Fl = mf_response2(S, 0, S['w0'][0], us, ndraw=ndraw, seed=net, mode=mode, sw=sw)
        _MF[key] = (F, Fl.var(0), Sb)
    return _MF[key]


def rmax_for(trig, kappa):
    if np.isnan(trig):
        return float(np.clip(1.2 * (US[-1] - 1) / kappa, 1.0, 40.0))
    return float(np.clip(1.6 * trig + 0.5, 1.0, 40.0))


def ramp(S, kappa, tau_m, rmax, md, seed, T2=20000, **kw):
    beta, tau_r = MODES[md]
    rext = np.concatenate([np.linspace(0, rmax, T2), np.linspace(rmax, 0, T2)])
    o = run2(S, beta, gam=GAM, tau_m=tau_m, tau_r=tau_r, kappa=kappa, T=2 * T2, burn=0, seed=seed,
             rho_ext=rext, **kw)
    return rext, o


KAPPAS = np.geomspace(0.03, 1.0, 8)
TMS = np.geomspace(1.0, 30.0, 8)


def f_task(args):
    net, kappa, tau_m, mode, rep = args
    F = mf_curve(net)[0]
    code, ptrig, prel = classify_regime(US, F, kappa, tau_m)
    rmax = rmax_for(ptrig if code in (1, 2) else np.nan, kappa)
    if mode == 'adaptive':
        rmax = min(40.0, 1.5 * rmax)
    S = base_state(net)
    rext, o = ramp(S, kappa, tau_m, rmax, mode, 3000 + 17 * net + rep)
    reg, trig, rel = detect(rext, o[2][:, 0], 20000)
    return dict(net=net, kappa=kappa, tau_m=tau_m, mode=mode, rep=rep, rmax=rmax, pred=REG[code],
                pred_trig=ptrig, pred_rel=prel, regime=reg, trig=trig, rel=rel,
                phi_end=o[2][-500:, 0].mean())


def c_ramp_task(args):
    net, kappa, tau_m, T2, rep = args
    F = mf_curve(net)[0]
    code, ptrig, prel = classify_regime(US, F, kappa, tau_m)
    rmax = rmax_for(ptrig if code in (1, 2) else np.nan, kappa)
    S = base_state(net)
    rext, o = ramp(S, kappa, tau_m, rmax, 'frozen', 8000 + rep, T2=T2)
    reg, trig, rel = detect(rext, o[2][:, 0], T2, k=max(50, T2 // 100))
    return dict(net=net, kappa=kappa, tau_m=tau_m, T2=T2, rep=rep, pred=REG[code], pred_trig=ptrig,
                pred_rel=prel, regime=reg, trig=trig, rel=rel)


def c_mf_task(args):
    net, ndraw, nus = args
    us = np.linspace(0.6, 4.0, nus)
    F = mf_curve(net, ndraw=ndraw, us=us)[0]
    rows = []
    for k in KAPPAS:
        for tm in TMS:
            code, tr, rl = classify_regime(us, F, k, tm)
            rows.append(dict(net=net, ndraw=ndraw, nus=nus, kappa=k, tau_m=tm, pred=REG[code], trig=tr, rel=rl))
    return rows


# ------------------------------------------------------------------ H: heterogeneous deterministic map
def h_task(args):
    net, kappa, tau_m = args
    S = base_state(net)
    lam = np.exp(-1.0 / tau_m)
    rng = np.random.default_rng(net)
    M = S['Xbar'].shape[1]
    anom = (S['Lch'][0] @ rng.standard_normal((M, 2000))).T.copy()
    idx = rng.integers(0, 2000, 300)
    w = np.ascontiguousarray(S['w0'][0])
    D, X, P, sl = S['Dbar'][0], S['Xbar'][0], S['Psc'][0], S['slack'][0]
    D0 = (w.T @ D)
    load = D0 / D0.sum()

    def fixed(r, rx, iters=600, tol=1e-7):
        for _ in range(iters):
            rn = lam * r + expected_failure(D, w, X, P, sl, anom, idx, r, kappa, rx, 0.5, 0.8, 0, 0.05)
            if np.max(np.abs(rn - r)) < tol:
                return rn
            r = rn
        return r

    F = mf_curve(net)[0]
    code, ptrig, prel = classify_regime(US, F, kappa, tau_m)
    rmax = rmax_for(ptrig if code in (1, 2) else np.nan, kappa)
    grid = np.linspace(0, rmax, 121)
    r = np.zeros(M)
    up = []
    for rx in grid:
        r = fixed(r, rx)
        up.append(load @ r * (1 - lam))
    dn = []
    for rx in grid[::-1]:
        r = fixed(r, rx)
        dn.append(load @ r * (1 - lam))
    up, dn = np.array(up), np.array(dn)[::-1]
    thr = 0.5 * (up.max() + up.min()) if up.max() - up.min() > 0.05 else np.inf
    hi_up = np.where(up > thr)[0]
    trig = grid[hi_up[0]] if len(hi_up) else np.nan
    hi_dn = np.where(dn > thr)[0]
    rel = grid[hi_dn[0]] if len(hi_dn) else np.nan
    gap = np.max(np.abs(up - dn))
    if up[0] > 0.5 * (up.max() + up.min()) and gap < 0.02:
        reg = 'prefrag'
    elif gap < 0.02:
        reg = 'mono'
    elif dn[0] > thr:
        reg = 'trap'
    else:
        reg = 'hyst'
    return dict(net=net, kappa=kappa, tau_m=tau_m, pred=REG[code], pred_trig=ptrig, pred_rel=prel,
                het_regime=reg, het_trig=trig, het_rel=rel, het_gap=gap, rmax=rmax,
                grid=grid.tolist(), F_up=up.tolist(), F_dn=dn.tolist())


# ------------------------------------------------------------------ R: residence times
def r_task(args):
    net, N, M, kappa, tau_m, rep = args
    S = base_state(net, N=N, M=M)
    T1, T3 = 20000, 100000
    F = mf_curve(net, N=N, M=M)[0]
    code, ptrig, prel = classify_regime(US, F, kappa, tau_m)
    rmax = rmax_for(ptrig if code in (1, 2) else np.nan, kappa)
    rext = np.concatenate([np.linspace(0, rmax, T1), np.zeros(T3)])
    o = run2(S, 0.0, gam=GAM, tau_m=tau_m, tau_r=1e9, kappa=kappa, T=T1 + T3, burn=0, seed=9000 + rep,
             rho_ext=rext)
    ph = np.convolve(o[2][:, 0], np.ones(200) / 200, 'same')
    collapsed = ph[T1 - 300:T1].mean() > 0.5
    post = ph[T1 + 200:]
    below = np.where(post < 0.5)[0]
    tres = float(below[0] + 200) if (collapsed and len(below)) else (np.nan if not collapsed else np.inf)
    return dict(net=net, N=N, M=M, kappa=kappa, tau_m=tau_m, rep=rep, pred=REG[code], pred_rel=prel,
                collapsed=bool(collapsed), residence=tres, censor=T3, phi_post_mean=post.mean())


# ------------------------------------------------------------------ A: alternative closures
CLOS = {'threshold': dict(mode=0), 'smooth_0.05': dict(mode=1, sw=0.05), 'smooth_0.2': dict(mode=1, sw=0.2),
        'rationing': dict(mode=2), 'bounded_0.5': dict(kmax=0.5), 'bounded_0.25': dict(kmax=0.25),
        'memoryless': dict(memoryless=True)}


def a_task(args):
    net, clo, kappa, tau_m, rep = args
    kw = dict(CLOS[clo])
    tm = 0.05 if kw.pop('memoryless', False) else tau_m
    S = base_state(net)
    rmax = float(min(40.0, 3.0 / kappa))
    rext, o = ramp(S, kappa, tm, rmax, 'frozen', 11000 + rep, **kw)
    T2 = 20000
    reg, trig, rel = detect(rext, o[2][:, 0], T2)
    md, sw = kw.get('mode', 0), kw.get('sw', 0.05)
    if 'kmax' not in kw:
        F = mf_curve(net, mode=md, sw=sw)[0]
        code, ptrig, prel = classify_regime(US, F, kappa, tm)
        pred = REG[code]
    else:
        pred, ptrig, prel = 'na', np.nan, np.nan
    sb = o[1][:, 0]
    return dict(net=net, closure=clo, kappa=kappa, tau_m=tau_m, rep=rep, rmax=rmax, regime=reg, trig=trig,
                rel=rel, pred=pred, pred_trig=ptrig, pred_rel=prel,
                width=(trig - (rel if np.isfinite(rel) else 0.0)) if np.isfinite(trig) else np.nan,
                sb_up=sb[:T2].mean(), sb_dn=sb[T2:].mean(), sb_max=np.convolve(sb, np.ones(200) / 200, 'same').max())


# ------------------------------------------------------------------ S: substitution
def chi_mat(kind):
    L = 4
    if kind == 'none':
        return np.zeros((L, L)), 0.0
    if kind.startswith('uniform'):
        v, cap = [float(x) for x in kind.split('_')[1:]]
        c = np.full((L, L), v)
        np.fill_diagonal(c, 0)
        return c, cap
    if kind.startswith('directed'):
        cap = float(kind.split('_')[1])
        c = np.zeros((L, L))
        c[0, 1] = c[1, 0] = 0.3   # wheat <-> maize (feed/food grain)
        c[0, 2] = c[2, 0] = 0.15  # wheat <-> rice (staple)
        c[1, 3] = 0.05            # maize -> soybean (feed ration)
        return c, cap


SUB = ['none', 'uniform_0.1_0.25', 'uniform_0.3_0.25', 'uniform_0.3_1.0', 'uniform_0.5_1.0', 'directed_0.25',
       'directed_1.0']


def s_task(args):
    kind, rep = args
    SM = stack_layers([base_state(n) for n in range(4)])
    chi, cap = chi_mat(kind)
    T2 = 20000
    rext = np.concatenate([np.linspace(0, 3.0, T2), np.linspace(3.0, 0, T2)])
    o = run2(SM, 0.0, gam=GAM, tau_m=5.0, tau_r=1e9, kappa=0.15, chi=chi, cap=cap, T=2 * T2, burn=0,
             seed=6000 + rep, rho_ext=rext)
    out = dict(kind=kind, rep=rep, cap=cap)
    trigs = []
    for l in range(4):
        reg, trig, rel = detect(rext, o[2][:, l], T2)
        out['reg%d' % l], out['trig%d' % l], out['rel%d' % l] = reg, trig, rel
        trigs.append(trig)
        out['sb%d' % l] = o[1][:T2, l].mean()
        out['so%d' % l] = o[0][:T2, l].mean()
    tr = np.array(trigs)
    out['trig_sd'] = np.nanstd(tr) if np.isfinite(tr).sum() > 1 else np.nan
    out['n_collapse'] = int(np.isfinite(tr).sum())
    k = 100
    ds = lambda x: x[:len(x) // k * k].reshape(-1, k).mean(1)
    for l in range(4):
        out['phi%d_ds' % l] = ds(o[2][:, l]).tolist()
    out['rext_ds'] = ds(rext).tolist()
    return out


# ------------------------------------------------------------------ E: EWS + innovation variance
def e_task(args):
    rep, kappa, tau_m, rstop = args
    S = base_state(0)
    T = 30000
    rext = np.linspace(0, rstop, T)
    o = run2(S, 0.0, gam=GAM, tau_m=tau_m, tau_r=1e9, kappa=kappa, T=T, burn=0, seed=5000 + rep, rho_ext=rext)
    from scipy.ndimage import gaussian_filter1d
    res = {}
    lam = np.exp(-1 / tau_m)
    sal = o[6][:, 0]
    for name, x in [('sal', sal), ('short', o[0][:, 0])]:
        resid = x - gaussian_filter1d(x, 300)
        W = 1500
        idx = np.arange(W, T, 250)
        res[name + '_ac1'] = np.array([np.corrcoef(resid[i - W:i - 1], resid[i - W + 1:i])[0, 1] for i in idx])
        res[name + '_var'] = np.array([resid[i - W:i].var() for i in idx])
    innov = sal[1:] - lam * sal[:-1]
    ri = innov - gaussian_filter1d(innov, 300)
    res['innov_var'] = np.array([ri[i - W:i].var() for i in idx])
    res['rext'] = rext[idx]
    res['u_mean'] = np.array([1 + kappa * (sal[i - W:i].mean() + rext[i - W:i].mean()) for i in idx])
    res['rep'] = rep
    return res


def b2_task(args):
    """Extension of the focal-importer grid beyond beta_i = 60 (same CRN seeds as b_task)."""
    global BI
    keep = BI
    BI = [60, 120, 240, 480]
    try:
        return b_task(args)
    finally:
        BI = keep


# ------------------------------------------------------------------ M: polymorphic (mixed) populations
def m_task(args):
    net, tau_m, f, bhigh, mu = args
    S = spatial_system(seed=net)
    N = S['Dbar'].shape[1]
    rng = np.random.default_rng(700 + net)
    order = rng.permutation(N)
    nh = int(round(f * N))
    high = np.zeros(N, bool)
    high[order[:nh]] = True
    bv = np.where(high, float(bhigh), 0.0)
    o = run2(S, bv, gam=GAM, tau_m=tau_m, tau_r=2.0, mu=mu, T=8000, burn=1000, seed=31000 + net)
    shi = o[8].mean(0)
    return dict(net=net, tau_m=tau_m, f=f, bhigh=bhigh, mu=mu, short_high=shi[high].mean() if nh else np.nan,
                short_zero=shi[~high].mean() if nh < N else np.nan, sys_short=o[1][:, 0].mean())


if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    blocks = sys.argv[2] if len(sys.argv) > 2 else 'PBFCHRASE'
    t0 = time.time()
    meta = {}
    _ = p_task(('base', 0, 2.0))
    with Pool(nproc) as pool:
        if 'F' in blocks or 'C' in blocks or 'H' in blocks or 'E' in blocks:
            mfs = pool.map(mf_curve, range(10))
            np.savez(f'{OUT}/F_mf.npz', us=US, F=np.array([m[0] for m in mfs]), Fvar=np.array([m[1] for m in mfs]),
                     Sb=np.array([m[2] for m in mfs]))
            print('mf', time.time() - t0, flush=True)
        if 'P' in blocks:
            tasks = [(v, n, b) for v in VARIANTS for n in range(20) for b in BETAS]
            pd.DataFrame(pool.map(p_task, tasks, chunksize=2)).to_csv(f'{OUT}/P_paradox.csv', index=False)
            print('P', time.time() - t0, flush=True)
        if 'B' in blocks:
            tasks = [(mu, tm, n, k, bp) for mu in (1.0, 0.0) for tm in (2.0, 5.0, 10.0) for n in range(10)
                     for k in range(2) for bp in BPOP]
            rows = [r for rr in pool.map(b_task, tasks, chunksize=1) for r in rr]
            pd.DataFrame(rows).to_csv(f'{OUT}/B_bestresponse.csv', index=False)
            print('B', time.time() - t0, flush=True)
        if 'F' in blocks:
            tasks = [(n, float(k), float(tm), md, rep) for n in range(10) for k in KAPPAS for tm in TMS
                     for md in ('frozen', 'adaptive') for rep in range(2)]
            pd.DataFrame(pool.map(f_task, tasks, chunksize=1)).to_csv(f'{OUT}/F_regimes.csv', index=False)
            print('F', time.time() - t0, flush=True)
        if 'C' in blocks:
            sub = [(float(KAPPAS[i]), float(TMS[j])) for i in (2, 4, 6) for j in (1, 3, 5, 7)]
            tasks = [(n, k, tm, T2, rep) for n in range(5) for (k, tm) in sub for T2 in (5000, 20000, 80000)
                     for rep in range(2)]
            pd.DataFrame(pool.map(c_ramp_task, tasks, chunksize=1)).to_csv(f'{OUT}/C_ramp.csv', index=False)
            tasks = [(n, nd, nu) for n in range(10) for nd in (200, 600, 2000) for nu in (86, 171, 341)]
            rows = [r for rr in pool.map(c_mf_task, tasks, chunksize=1) for r in rr]
            pd.DataFrame(rows).to_csv(f'{OUT}/C_mf.csv', index=False)
            print('C', time.time() - t0, flush=True)
        if 'H' in blocks:
            tasks = [(n, float(KAPPAS[i]), float(TMS[j])) for n in range(5) for i in (1, 3, 5, 7) for j in (0, 2, 4, 6)]
            res = pool.map(h_task, tasks, chunksize=1)
            json.dump(res, open(f'{OUT}/H_hetmap.json', 'w'), default=float)
            pd.DataFrame([{k: v for k, v in r.items() if k not in ('grid', 'F_up', 'F_dn')} for r in res]).to_csv(
                f'{OUT}/H_hetmap.csv', index=False)
            print('H', time.time() - t0, flush=True)
        if 'R' in blocks:
            tasks = [(n, N, M, k, tm, rep) for n in range(5) for (N, M) in ((50, 15), (100, 30), (200, 60))
                     for (k, tm) in ((0.3, 10.0), (0.5, 20.0), (0.15, 5.0)) for rep in range(2)]
            pd.DataFrame(pool.map(r_task, tasks, chunksize=1)).to_csv(f'{OUT}/R_residence.csv', index=False)
            print('R', time.time() - t0, flush=True)
        if 'A' in blocks:
            tasks = [(n, c, k, tm, rep) for n in range(5) for c in CLOS for k in (0.1, 0.2, 0.4) for tm in (3.0, 10.0)
                     for rep in range(2)]
            pd.DataFrame(pool.map(a_task, tasks, chunksize=1)).to_csv(f'{OUT}/A_closures.csv', index=False)
            print('A', time.time() - t0, flush=True)
        if 'S' in blocks:
            tasks = [(k, rep) for k in SUB for rep in range(4)]
            res = pool.map(s_task, tasks, chunksize=1)
            json.dump(res, open(f'{OUT}/S_subst.json', 'w'))
            pd.DataFrame([{k: v for k, v in r.items() if not k.endswith('_ds')} for r in res]).to_csv(
                f'{OUT}/S_subst.csv', index=False)
            print('S', time.time() - t0, flush=True)
        if 'Q' in blocks:
            tasks = [(1.0, tm, n, k, bp) for tm in (2.0, 5.0, 10.0) for n in range(10) for k in range(2)
                     for bp in (0, 1, 2, 3, 4)]
            rows = [r for rr in pool.map(b2_task, tasks, chunksize=1) for r in rr]
            pd.DataFrame(rows).to_csv(f'{OUT}/Q_bestresponse_ext.csv', index=False)
            print('Q', time.time() - t0, flush=True)
        if 'M' in blocks:
            tasks = [(n, tm, f, bh, mu) for n in range(10) for tm in (2.0, 5.0, 10.0)
                     for f in np.round(np.linspace(0, 1, 11), 2) for bh in (10.0, 20.0, 60.0) for mu in (1.0,)]
            pd.DataFrame(pool.map(m_task, tasks, chunksize=1)).to_csv(f'{OUT}/M_mixture.csv', index=False)
            print('M', time.time() - t0, flush=True)
        if 'E' in blocks:
            F = mf_curve(0)[0]
            rext_, rho_, gain_, folds_ = mf_branches_exact(US, F, 0.15, 5.0)
            code, ptrig, prel = classify_regime(US, F, 0.15, 5.0)
            rstop = 0.97 * ptrig if np.isfinite(ptrig) and ptrig > 0 else 1.0
            res = pool.map(e_task, [(rep, 0.15, 5.0, rstop) for rep in range(48)])
            np.save(f'{OUT}/E_ews.npy', np.array(res, dtype=object), allow_pickle=True)
            meta['E_rstop'] = rstop
            meta['E_ptrig'] = ptrig
            print('E', time.time() - t0, flush=True)
    meta['runtime_s'] = time.time() - t0
    json.dump(meta, open(f'{OUT}/meta.json', 'w'))
