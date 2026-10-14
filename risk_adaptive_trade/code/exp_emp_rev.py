"""Revision: model-based analyses on the empirical FAOSTAT multiplex (wheat, maize, rice, soybean; 1986-2024).
Usage: python3 exp_emp_rev.py <nproc> [blocks]      panels are read from ./panels/emp_<spec>.npz
Blocks  S  static stress scans per layer-year (extended surge grid u in [1,4]; parameter grid; panel variants)
        X  factorial decomposition of the change since 1986-88 (structure / shock exposure / margins), paired draws
        D  risk-aversion sweeps on the 2006 and 2019 topologies (public and private salience)
        L  hysteresis loops on the 2006 / 2019 multiplex, and held-out layer-year closure validation
        V  event-study benchmark: model replay of the historical anomaly sequence (supply-only vs memory)
"""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_core2 import simulate2, static_scan2
from systems2 import mf_branches_exact, classify_regime

OUT = 'out_emp_rev'
os.makedirs(OUT, exist_ok=True)
US = np.round(np.linspace(1.0, 4.0, 61), 3)
EPS0 = 0.2
DEFAULT = 'mirror_raw_lowess30'
_P = {}


def panel(spec):
    if spec not in _P:
        d = np.load(f'panels/emp_{spec}.npz', allow_pickle=True)
        _P[spec] = dict(FL=d['flows'].astype(float), PR=d['prod'].astype(float), AN=d['anom'].astype(float),
                        Z=d['z'], YEARS=d['years'])
    return _P[spec]


def layer_year(spec, l, t, eps, smooth=1, margin='uniform', t_struct=None, t_prod=None, t_marg=None):
    """Static inputs for layer l.  Structure (shares, demand, exports) from window around t_struct, shock
    exposure (production) around t_prod, margins around t_marg (defaults t).  margin: 'uniform' eps*max(P,E);
    'revealed' = demonstrated surge capacity max(E over previous 5 years) - E, plus 0.1 E;
    'relative' = eps*max(P,E) evaluated in the margin year as a RATIO to E and re-applied to current E."""
    P_ = panel(spec)
    FL, PR = P_['FL'], P_['PR']
    NY = FL.shape[1]
    ts = t if t_struct is None else t_struct
    tp = t if t_prod is None else t_prod
    tm = t if t_marg is None else t_marg
    win = lambda x: slice(max(0, x - smooth), min(NY, x + smooth + 1))
    Fm = FL[l, win(ts)].mean(0)
    D = Fm.sum(1)
    E = Fm.sum(0)
    w = np.divide(Fm, D[:, None], out=np.zeros_like(Fm), where=D[:, None] > 0)
    P = PR[l, win(tp)].mean(0)
    if margin == 'uniform':
        Pm = PR[l, win(tm)].mean(0)
        Em = FL[l, win(tm)].mean(0).sum(0)
        slack = eps * np.maximum(Pm, Em)
    elif margin == 'relative':
        Pm = PR[l, win(tm)].mean(0)
        Em = FL[l, win(tm)].mean(0).sum(0)
        ratio = np.divide(eps * np.maximum(Pm, Em), Em, out=np.full_like(Em, eps), where=Em > 0)
        slack = ratio * np.maximum(E, 1e-9)
        slack = np.where(Em > 0, slack, eps * np.maximum(P, E))
    else:  # revealed surge capacity
        hist = FL[l, max(0, tm - 5):tm + 1].sum(1)        # (years, K) exports
        Em = FL[l, win(tm)].mean(0).sum(0)
        slack = np.maximum(hist.max(0) - Em, 0.0) + 0.1 * Em
    return Fm, D, E, w, P, slack


def scan(spec, l, t, eps, b, q, smooth=1, margin='uniform', seed=None, ndraw=400, **kw):
    Fm, D, E, w, P, slack = layer_year(spec, l, t, eps, smooth, margin, **kw)
    AN = panel(spec)['AN']
    ii = np.where(D > 0)[0]
    jj = np.where(E > 0)[0]
    sd = (1000 + 7 * t + l) if seed is None else seed
    Sh, Sb, Ph, Fl = static_scan2(D[ii], np.ascontiguousarray(w[np.ix_(ii, jj)]), E[jj], P[jj], slack[jj],
                                  np.ascontiguousarray(AN[l][:, jj]), US, ndraw, sd, b, q, 0, 0.05)
    return Sh, Sb, Ph, Fl, D[ii], w[np.ix_(ii, jj)], E[jj]


def u_cross(curve, thr=0.5):
    c = np.asarray(curve)
    i = np.where(c >= thr)[0]
    if len(i) == 0:
        return np.nan   # censored: not reached by u = 4
    i = i[0]
    if i == 0:
        return 1.0
    return US[i - 1] + (thr - c[i - 1]) / (c[i] - c[i - 1]) * (US[i] - US[i - 1])


def summarize(Sh, Sb, Ph, Fl):
    sb1 = Sb[:, 0]
    qq = np.quantile(sb1, 0.95)
    i12 = int(np.argmin(np.abs(US - 1.2)))
    return dict(h50=u_cross(Ph.mean(0), 0.5), h25=u_cross(Ph.mean(0), 0.25), phi1=Ph[:, 0].mean(),
                sb1_mean=sb1.mean(), sb1_cvar95=sb1[sb1 >= qq].mean(), so1_mean=Sh[:, 0].mean(),
                sb12_mean=Sb[:, i12].mean(), so12_mean=Sh[:, i12].mean(), p_frag1=(Ph[:, 0] > 0.25).mean())


def s_task(args):
    spec, l, t, eps, b, q, smooth, margin = args
    Sh, Sb, Ph, Fl, D, w, E = scan(spec, l, t, eps, b, q, smooth, margin)
    yrs = panel(spec)['YEARS']
    hhi = np.average((w ** 2).sum(1), weights=D)
    exs = E / E.sum()
    r = dict(spec=spec, layer=l, year=int(yrs[t]), eps=eps, b=b, q=q, smooth=smooth, margin=margin,
             n_imp=len(D), n_exp=len(E), hhi_imp=hhi, hhi_exp=(exs ** 2).sum(), **summarize(Sh, Sb, Ph, Fl))
    if spec == DEFAULT and eps == EPS0 and b == 0.5 and q == 0.8 and smooth == 1 and margin == 'uniform':
        r.update(F_curve=Fl.mean(0).tolist(), Ph_curve=Ph.mean(0).tolist(), Sb_curve=Sb.mean(0).tolist(),
                 Sh_curve=Sh.mean(0).tolist())
    return r


def x_task(args):
    """2^3 factorial: structure s, exposure p, margins m each at base (1987 = 1986-88 window) or year t;
    identical anomaly draws (paired) across corners."""
    l, t, margin = args
    tb = 1
    out = dict(layer=l, year=int(panel(DEFAULT)['YEARS'][t]), margin=margin)
    for s_ in (0, 1):
        for p_ in (0, 1):
            for m_ in (0, 1):
                Sh, Sb, Ph, Fl, *_ = scan(DEFAULT, l, t, EPS0, 0.5, 0.8, 1, margin, seed=777 + l,
                                          t_struct=t if s_ else tb, t_prod=t if p_ else tb, t_marg=t if m_ else tb)
                sm = summarize(Sh, Sb, Ph, Fl)
                key = f's{s_}p{p_}m{m_}'
                out[key + '_h50'] = sm['h50']
                out[key + '_h25'] = sm['h25']
                out[key + '_sb12'] = sm['sb12_mean']
                out[key + '_phi1'] = sm['phi1']
    return out


# ------------------------------------------------------------------ dynamics on the empirical multiplex
def multiplex_state(spec, t, eps, layers=(0, 1, 2, 3), floor=1e-3, window=4):
    P_ = panel(spec)
    FL, AN = P_['FL'], P_['AN']
    K = FL.shape[2]
    Ls = len(layers)
    S = {k: np.zeros((Ls, K, K)) for k in ('A', 'C', 'w0')}
    for k in ('Dbar', 'Xbar', 'slack', 'Psc'):
        S[k] = np.zeros((Ls, K))
    S['anom'] = np.zeros((Ls, AN.shape[1], K))
    for a, l in enumerate(layers):
        Fm, D, E, w, P, sl = layer_year(spec, l, t, eps)
        acc = (FL[l, max(0, t - window):t + 1].sum(0) > 0) & (D[:, None] > 0)
        ww = np.where(acc, np.maximum(w, floor), 0.0)
        ww = np.divide(ww, ww.sum(1, keepdims=True), out=np.zeros_like(ww), where=ww.sum(1, keepdims=True) > 0)
        S['A'][a], S['w0'][a] = acc, ww
        S['C'][a] = np.where(acc, -np.log(np.maximum(ww, 1e-12)), 0.0)
        S['Dbar'][a], S['Xbar'][a], S['slack'][a], S['Psc'][a] = D, E, sl, P
        S['anom'][a] = AN[l]
    S['Lch'] = np.tile(np.eye(K)[None], (Ls, 1, 1))
    return S


def run(S, beta, tau_m, tau_r, kappa=0.0, chi=None, cap=0.0, T=4000, burn=500, seed=1, rho_ext=None, mu=1.0,
        b=0.5, q=0.8, shock_mode=0, rec=-1, anom=None):
    Ls, K, _ = S['A'].shape
    rx = np.zeros(T) if rho_ext is None else rho_ext
    chi = np.zeros((Ls, Ls)) if chi is None else chi
    lam = np.exp(-1.0 / tau_m)
    eta = 1.0 - np.exp(-1.0 / tau_r)
    an = S['anom'] if anom is None else anom
    return simulate2(S['A'], S['C'], S['Dbar'], S['Xbar'], S['slack'], S['Psc'], S['Lch'], an, S['w0'],
                     np.full(K, float(beta)), np.full(K, lam), np.full(K, eta), lam, float(mu), 0.0,
                     1.0, b, q, 0, 0.05, float(kappa), 0.0, chi, float(cap), 0.0, 0.0, int(shock_mode),
                     np.zeros((Ls, K)), rx, int(T), int(burn), int(seed), int(rec))


def d_task(args):
    year, layers, beta, tau_m, mu, seed = args
    P_ = panel(DEFAULT)
    t = int(np.where(P_['YEARS'] == year)[0][0])
    S = multiplex_state(DEFAULT, t, EPS0, layers=layers)
    o = run(S, beta, tau_m, 2.0, T=4000, burn=500, seed=seed, mu=mu)
    Dw = S['Dbar'].sum(1)
    sb = (o[1] * Dw[None, :]).sum(1) / Dw.sum()
    so = (o[0] * Dw[None, :]).sum(1) / Dw.sum()
    qq = np.quantile(sb, 0.95)
    return dict(year=year, layers='-'.join(map(str, layers)), beta=beta, tau_m=tau_m, mu=mu, seed=seed,
                sb_mean=sb.mean(), sb_cvar95=sb[sb >= qq].mean(), so_mean=so.mean(), hhi_imp=o[5].mean(),
                hhi_agg=o[4].mean(), nres=o[3].sum(1).mean(), total_demand=float(Dw.sum()),
                by_layer_sb=o[1].mean(0).tolist())


def ds(x, k=100):
    if x.ndim == 1:
        return x[:len(x) // k * k].reshape(-1, k).mean(1)
    return x[:len(x) // k * k].reshape(-1, k, x.shape[1]).mean(1)


def detect(rext, phi, T2, thr=0.5, k=200):
    ph = np.convolve(phi, np.ones(k) / k, 'same')
    up, dn = ph[:T2], ph[T2:]
    if up[k:2 * k].mean() > thr:
        return 'prefrag', np.nan, np.nan
    iu = np.where(up > thr)[0]
    if not len(iu):
        return 'mono', np.nan, np.nan
    trig = rext[:T2][iu[0]]
    above = np.where(dn > thr)[0]
    if len(above) == 0:
        return 'hyst', trig, rext[T2:][0]
    if above[-1] >= len(dn) - k:
        return 'trap', trig, np.nan
    return 'hyst', trig, rext[T2:][above[-1]]


def l_task(args):
    year, kappa, tau_m, chi, rmax, seed, mode = args
    P_ = panel(DEFAULT)
    t = int(np.where(P_['YEARS'] == year)[0][0])
    S = multiplex_state(DEFAULT, t, EPS0)
    T2 = 12000
    rext = np.concatenate([np.linspace(0, rmax, T2), np.linspace(rmax, 0, T2)])
    beta, tau_r = (0.0, 1e9) if mode == 'frozen' else (2.0, 2.0)
    chim = np.full((4, 4), chi)
    np.fill_diagonal(chim, 0.0)
    o = run(S, beta, tau_m, tau_r, kappa=kappa, chi=chim, cap=0.25, T=2 * T2, burn=0, seed=seed, rho_ext=rext)
    regs = [detect(rext, o[2][:, l], T2) for l in range(4)]
    return dict(year=year, kappa=kappa, tau_m=tau_m, chi=chi, rmax=rmax, seed=seed, mode=mode,
                rext=ds(rext).tolist(), phi=ds(o[2]).tolist(), so=ds(o[0]).tolist(), sb=ds(o[1]).tolist(),
                regimes=[r[0] for r in regs], trig=[float(r[1]) for r in regs], rel=[float(r[2]) for r in regs])


def h_task(args):
    """Held-out validation of the scalar closure on single empirical layers."""
    year, l, kappa, tau_m = args
    P_ = panel(DEFAULT)
    t = int(np.where(P_['YEARS'] == year)[0][0])
    S = multiplex_state(DEFAULT, t, EPS0, layers=(l,))
    D, w, X, Pp, sl = S['Dbar'][0], S['w0'][0], S['Xbar'][0], S['Psc'][0], S['slack'][0]
    ii = np.where(D > 0)[0]
    jj = np.where(X > 0)[0]
    usg = np.linspace(0.6, 4.0, 171)
    Sh, Sb, Ph, Fl = static_scan2(D[ii], np.ascontiguousarray(w[np.ix_(ii, jj)]), X[jj], Pp[jj], sl[jj],
                                  np.ascontiguousarray(P_['AN'][l][:, jj]), usg, 400, 31 + l, 0.5, 0.8, 0, 0.05)
    F = Fl.mean(0)
    code, ptrig, prel = classify_regime(usg, F, kappa, tau_m)
    rmax = float(np.clip(1.6 * ptrig + 0.5, 1.0, 40.0)) if code in (1, 2) else float(
        np.clip(1.2 * 3.0 / kappa, 1.0, 40.0))
    T2 = 12000
    rext = np.concatenate([np.linspace(0, rmax, T2), np.linspace(rmax, 0, T2)])
    o = run(S, 0.0, tau_m, 1e9, kappa=kappa, T=2 * T2, burn=0, seed=77, rho_ext=rext)
    reg, trig, rel = detect(rext, o[2][:, 0], T2)
    return dict(year=year, layer=l, kappa=kappa, tau_m=tau_m, pred={0: 'mono', 1: 'hyst', 2: 'trap', 3: 'prefrag'}[code],
                pred_trig=ptrig, pred_rel=prel, regime=reg, trig=trig, rel=rel, rmax=rmax)


def v_task(args):
    """Replay the historical anomaly sequence 1986-2024 on the fixed 2006 network of one layer (annual steps),
    after a burn-in of 40 bootstrapped years; returns simulated layer flows for the event-study estimator."""
    l, beta, tau_m, rep = args
    P_ = panel(DEFAULT)
    t = int(np.where(P_['YEARS'] == 2006)[0][0])
    S = multiplex_state(DEFAULT, t, EPS0, layers=(l,))
    NY = P_['AN'].shape[1]
    burn = 40
    rng = np.random.default_rng(100 * l + rep)
    seq = np.concatenate([rng.integers(0, NY, burn), np.arange(NY)])
    an = np.ascontiguousarray(S['anom'][:, seq, :])
    o = run(S, beta, tau_m, 1.0, T=len(seq), burn=burn, seed=rep, shock_mode=2, rec=-2, anom=an)
    import eventstudy as ES
    Sv, imps, msh = ES.shares_from_flows(o[12])
    E = ES.build_events(l, Sv, P_['Z'][l].T, msh, P_['YEARS'])
    rows = []
    for oc in ('k', 'a'):
        r, _ = ES.estimate(E, oc, nboot=200, seed=rep, by_layer=False)
        if r is not None:
            r['outcome'] = oc
            rows.append(r)
    R = pd.concat(rows) if rows else pd.DataFrame()
    R['layer'], R['beta'], R['tau_m'], R['rep'] = l, beta, tau_m, rep
    sal = o[10]
    return R


if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    blocks = sys.argv[2] if len(sys.argv) > 2 else 'SXDLHV'
    t0 = time.time()
    NY = panel(DEFAULT)['FL'].shape[1]
    _ = s_task((DEFAULT, 0, 20, EPS0, 0.5, 0.8, 1, 'uniform'))
    specs = sorted(f[4:-4] for f in os.listdir('panels') if f.startswith('emp_'))
    with Pool(nproc) as pool:
        if 'S' in blocks:
            grid = [(DEFAULT, l, t, e, b, q, sm, mg) for l in range(4) for t in range(NY)
                    for e in (0.1, 0.15, 0.2, 0.3) for (b, q) in ((0.5, 0.8), (1.0, 0.5), (0.25, 1.0))
                    for sm in (0, 1, 2) for mg in ('uniform', 'revealed')]
            grid += [(sp, l, t, EPS0, 0.5, 0.8, 1, 'uniform') for sp in specs if sp != DEFAULT
                     for l in range(4) for t in range(NY)]
            pd.DataFrame(pool.map(s_task, grid, chunksize=4)).to_json(f'{OUT}/S_static.json', orient='records')
            print('S', time.time() - t0, flush=True)
        if 'X' in blocks:
            tasks = [(l, t, mg) for l in range(4) for t in range(NY) for mg in ('uniform', 'relative')]
            pd.DataFrame(pool.map(x_task, tasks, chunksize=1)).to_csv(f'{OUT}/X_factorial.csv', index=False)
            print('X', time.time() - t0, flush=True)
        if 'D' in blocks:
            BET = [0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80]
            tasks = [(y, ly, b, tm, mu, s) for y in (2006, 2019) for ly in ((0,), (1,), (2,), (3,), (0, 1, 2, 3))
                     for b in BET for tm in (2.0, 5.0) for mu in (1.0, 0.0) for s in range(3)]
            pd.DataFrame(pool.map(d_task, tasks, chunksize=1)).to_json(f'{OUT}/D_beta.json', orient='records')
            print('D', time.time() - t0, flush=True)
        if 'L' in blocks:
            tasks = [(2006, k, 3.0, chi, 4.0, s, md) for k in (0.1, 0.2) for chi in (0.0, 0.3) for s in range(3)
                     for md in ('frozen', 'adaptive')]
            tasks += [(2019, k, 3.0, chi, 4.0, s, 'frozen') for k in (0.1, 0.2) for chi in (0.0, 0.3) for s in range(2)]
            pd.DataFrame(pool.map(l_task, tasks, chunksize=1)).to_json(f'{OUT}/L_loops.json', orient='records')
            print('L', time.time() - t0, flush=True)
        if 'H' in blocks:
            tasks = [(y, l, k, tm) for y in (1990, 2000, 2010, 2020) for l in range(4) for k in (0.1, 0.2, 0.4)
                     for tm in (2.0, 5.0)]
            pd.DataFrame(pool.map(h_task, tasks, chunksize=1)).to_csv(f'{OUT}/H_heldout.csv', index=False)
            print('H', time.time() - t0, flush=True)
        if 'V' in blocks:
            cfg = [(0.0, 1.0)] + [(b, tm) for b in (2.0, 5.0, 10.0) for tm in (0.05, 1.0, 2.0, 5.0)]
            tasks = [(l, b, tm, rep) for l in range(4) for (b, tm) in cfg for rep in range(10)]
            res = pool.map(v_task, tasks, chunksize=1)
            pd.concat(res, ignore_index=True).to_csv(f'{OUT}/V_replay_eventstudy.csv', index=False)
            print('V', time.time() - t0, flush=True)
    json.dump(dict(runtime_s=time.time() - t0), open(f'{OUT}/meta.json', 'w'))
