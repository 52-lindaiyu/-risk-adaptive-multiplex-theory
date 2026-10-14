"""Revision: empirical event study (descriptive share response to exporter yield shocks), sensitivity grid,
placebo-based size calibration, and separated portfolio-overlap metrics.
Inputs : handoff/emp_<spec>.npz (from build_panel.py)
Outputs: res/emp_rev/EV_main.csv, EV_sensitivity.csv, EV_placebo.csv, PORT_metrics.csv
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eventstudy as ES

OUT = 'res/emp_rev'
os.makedirs(OUT, exist_ok=True)
LAY = ['wheat', 'maize', 'rice', 'soybean']


def load(spec='mirror_raw_lowess30'):
    d = np.load(f'handoff/emp_{spec}.npz', allow_pickle=True)
    return d['flows'].astype(float), d['z'], d['years'], d['anom'].astype(float)


def events(FL, Z, YRS, lag=0, base_min=0.05, clean=3):
    Es = []
    for l in range(4):
        S, imps, msh = ES.shares_from_flows(FL[l])
        z = Z[l].T.copy()
        if lag:
            z = np.concatenate([np.full((z.shape[0], lag), np.nan), z[:, :-lag]], axis=1)
        Es.append(ES.build_events(LAY[l], S, z, msh, YRS, base_min=base_min, clean=clean))
    return pd.concat(Es, ignore_index=True)


if __name__ == '__main__':
    FL, Z, YRS, AN = load()
    E = events(FL, Z, YRS)
    res = []
    for oc in ('k', 'a'):
        r, _ = ES.estimate(E, oc, nboot=2000, seed=11)
        r['outcome'] = oc
        res.append(r)
    pd.concat(res).to_csv(f'{OUT}/EV_main.csv', index=False)
    # sensitivity grid
    rows = []
    for spec in ('mirror_raw_lowess30', 'imponly_raw_lowess30', 'exponly_raw_lowess30', 'mirror_harm_lowess30',
                 'mirror_raw_lowess20', 'mirror_raw_lowess50', 'mirror_raw_quad'):
        FLs, Zs, Ys, _ = load(spec)
        for lag in (0, 1):
            for base_min in (0.02, 0.05, 0.10):
                for clean in (1, 3):
                    if spec != 'mirror_raw_lowess30' and not (base_min == 0.05 and clean == 3):
                        continue
                    Ex = events(FLs, Zs, Ys, lag=lag, base_min=base_min, clean=clean)
                    for oc in ('k', 'a'):
                        r, _ = ES.estimate(Ex, oc, nboot=500, seed=3, by_layer=False)
                        if r is None:
                            continue
                        r = r.set_index('k')
                        rows.append(dict(spec=spec, lag=lag, base_min=base_min, clean=clean, outcome=oc,
                                         n_events=int(r.n_events.iloc[0]), pretrend_p=r.pretrend_p.iloc[0],
                                         **{f'est_k{k}': r.loc[k, 'est'] for k in (-3, -2, 0, 1, 2, 3)},
                                         **{f'lo_k{k}': r.loc[k, 'lo'] for k in (0, 1, 2)},
                                         **{f'hi_k{k}': r.loc[k, 'hi'] for k in (0, 1, 2)}))
    pd.DataFrame(rows).to_csv(f'{OUT}/EV_sensitivity.csv', index=False)
    # placebo: permute each exporter's shock years (keeps shock frequency per exporter), 200 draws
    rng = np.random.default_rng(2026)
    prow = []
    for b in range(200):
        Zp = Z.copy()
        for l in range(4):
            for j in range(Z.shape[2]):
                Zp[l, :, j] = Z[l, rng.permutation(Z.shape[1]), j]
        Ep = events(FL, Zp, YRS)
        for oc in ('k', 'a'):
            r, _ = ES.estimate(Ep, oc, nboot=200, seed=b, by_layer=False)
            if r is None:
                continue
            r = r.set_index('k')
            prow.append(dict(draw=b, outcome=oc, **{f'est_k{k}': r.loc[k, 'est'] for k in (-3, -2, 0, 1, 2, 3)},
                             **{f'p_k{k}': r.loc[k, 'p_two'] for k in (0, 1, 2)}, pretrend_p=r.pretrend_p.iloc[0]))
    pd.DataFrame(prow).to_csv(f'{OUT}/EV_placebo.csv', index=False)
    # portfolio metrics, separated: w'Cw (total), HHI (diagonal), off-diagonal part, distinct-pair average corr
    rows = []
    for l in range(4):
        Cm = np.nan_to_num(np.corrcoef(AN[l].T))
        np.fill_diagonal(Cm, 1.0)
        for t in range(len(YRS)):
            Fm = FL[l, max(0, t - 1):min(len(YRS), t + 2)].mean(0)
            D = Fm.sum(1)
            ok = D > 0
            w = Fm[ok] / D[ok, None]
            tot = np.einsum('ij,jk,ik->i', w, Cm, w)
            hh = (w ** 2).sum(1)
            off = tot - hh
            denom = 1 - hh
            pair = np.divide(off, denom, out=np.full_like(off, np.nan), where=denom > 1e-9)
            rows.append(dict(layer=LAY[l], year=int(YRS[t]), wCw=np.average(tot, weights=D[ok]),
                             hhi=np.average(hh, weights=D[ok]), offdiag=np.average(off, weights=D[ok]),
                             pair_corr=np.nansum(pair * D[ok]) / D[ok][np.isfinite(pair)].sum()))
    pd.DataFrame(rows).to_csv(f'{OUT}/PORT_metrics.csv', index=False)
